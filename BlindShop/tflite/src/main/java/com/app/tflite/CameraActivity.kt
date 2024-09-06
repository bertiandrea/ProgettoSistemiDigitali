package com.app.tflite

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.util.Size
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.app.tflite.databinding.ActivityCameraBinding
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.random.Random


/** Activity that displays the camera and performs object detection on the incoming frames */
class CameraActivity : AppCompatActivity() {
    private lateinit var activityCameraBinding: ActivityCameraBinding
    private lateinit var bitmapBuffer: Bitmap
    private lateinit var textToSpeech: TextToSpeech

    private val executor = Executors.newSingleThreadExecutor()
    private val permissions = listOf(Manifest.permission.CAMERA)
    private val permissionsRequestCode = Random.nextInt(0, 10000)

    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK
    private val isFrontFacing get() = (lensFacing == CameraSelector.LENS_FACING_FRONT)

    private var pauseAnalysis = false
    private var imageRotationDegrees: Int = 0
    private val tfImageBuffer =
        TensorImage(DataType.FLOAT32) //DataType.FLOAT32 or DataType.UINT8 depending on MODEL

    private val tfImageProcessor by lazy {
        val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
        ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(
                ResizeOp(
                    tfInputSize.height,
                    tfInputSize.width,
                    ResizeOp.ResizeMethod.NEAREST_NEIGHBOR
                )
            )
            .add(Rot90Op(-imageRotationDegrees / 90))
            .add(NormalizeOp(127.5f, 127.5f))
            .build()
    }
    private val nnApiDelegate by lazy {
        NnApiDelegate()
    }
    private val tflite by lazy {
        Interpreter(
            FileUtil.loadMappedFile(this, MODEL_PATH),
            Interpreter.Options().addDelegate(nnApiDelegate)
        )
    }
    private val detector by lazy {
        ObjectDetectionHelper(tflite, FileUtil.loadLabels(this, LABELS_PATH))
    }
    private val tfInputSize by lazy {
        val inputIndex = 0
        val inputShape = tflite.getInputTensor(inputIndex).shape()
        Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityCameraBinding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(activityCameraBinding.root)
        //////////////////////////////////////////////////////////////////////////////
        // Inizializzazione di TextToSpeech
        textToSpeech = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.language = Locale.ITALIAN
            } else {
                Log.e(TAG, "TextToSpeech initialization failed")
            }
        }
        //////////////////////////////////////////////////////////////////////////////
        activityCameraBinding.fullscreenTouchCapture.setOnClickListener { it ->
            // Disable all camera controls
            it.isEnabled = false
            if (pauseAnalysis) {
                textToSpeech.speak("Analizzo", TextToSpeech.QUEUE_FLUSH, null, null)
                // If image analysis is in paused state, resume it
                pauseAnalysis = false
                activityCameraBinding.imagePredicted.visibility = View.GONE
            } else {
                // Otherwise, pause image analysis and freeze image
                pauseAnalysis = true
                /////////////////////////////////////////////////////////////////////////////////
                val tfImage =
                    tfImageProcessor.process(tfImageBuffer.apply { load(bitmapBuffer) }) //Shared Buffer
                val predictions = synchronized(detector) { detector.predict(tfImage) }
                val bestPrediction = predictions.maxByOrNull { it.score }
                reportPrediction(bestPrediction)
                speakOut(bestPrediction)
                if (TEST) {
                    val bitmapImage = testModelWithImage()
                    activityCameraBinding.imagePredicted.setImageBitmap(bitmapImage)
                    activityCameraBinding.imagePredicted.visibility = View.VISIBLE
                } else {
                    if (SHOW_PROCESSED_IMAGE) {
                        val bitmapProcessed = convertTfImageToBitmap(tfImage)
                        activityCameraBinding.imagePredicted.setImageBitmap(bitmapProcessed)
                        activityCameraBinding.imagePredicted.visibility = View.VISIBLE
                    } else {
                        val bitmapRotated =
                            rotateAndMirrorBitmap(bitmapBuffer, imageRotationDegrees, isFrontFacing)
                        activityCameraBinding.imagePredicted.setImageBitmap(bitmapRotated)
                        activityCameraBinding.imagePredicted.visibility = View.VISIBLE
                    }
                }
            }
            // Re-enable camera controls
            it.isEnabled = true
        }
    }

    private fun convertTfImageToBitmap(tfImage: TensorImage): Bitmap {
        val width = tfImage.width
        val height = tfImage.height
        val pixelData = FloatArray(width * height * 3) // RGB
        tfImage.tensorBuffer.floatArray.copyInto(pixelData)

        // Convert float RGB data to ARGB_8888 format
        val pixels = IntArray(width * height)
        for (i in pixels.indices) {
            val r = (((pixelData[i * 3] + 1) / 2) * 255).toInt().coerceIn(0, 255)
            val g = (((pixelData[i * 3 + 1] + 1) / 2) * 255).toInt().coerceIn(0, 255)
            val b = (((pixelData[i * 3 + 2] + 1) / 2) * 255).toInt().coerceIn(0, 255)
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b // ARGB
        }

        val bitmapProcessed = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmapProcessed.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmapProcessed
    }

    private fun rotateAndMirrorBitmap(
        bitmap: Bitmap,
        rotationDegrees: Int,
        isFrontFacing: Boolean
    ): Bitmap {
        val matrix = Matrix().apply {
            postRotate(rotationDegrees.toFloat())
            if (isFrontFacing) postScale(-1f, 1f)
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    @SuppressLint("SetTextI18n")
    private fun reportPrediction(prediction: ObjectDetectionHelper.ObjectPrediction?) =
        activityCameraBinding.viewFinder.post {
            if (prediction == null || prediction.score < ACCURACY_THRESHOLD) {
                activityCameraBinding.textPrediction.visibility = View.GONE
                return@post
            }
            activityCameraBinding.textPrediction.visibility = View.VISIBLE
            activityCameraBinding.textPrediction.text =
                "${"%.0f".format(prediction.score * 100)}%\n${prediction.label}"
        }

    private fun speakOut(prediction: ObjectDetectionHelper.ObjectPrediction?) {
        if (prediction == null || prediction.score < ACCURACY_THRESHOLD) {
            textToSpeech.speak("Sconosciuto", TextToSpeech.QUEUE_FLUSH, null, null)
        } else {
            textToSpeech.speak(prediction.label, TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }

    private fun testModelWithImage(): Bitmap {
        val inputStream = assets.open("Banana.jpg")
        val bitmapBuffer = BitmapFactory.decodeStream(inputStream)
        val tfImage = tfImageProcessor.process(tfImageBuffer.apply { load(bitmapBuffer) })
        val predictions = synchronized(detector) { detector.predict(tfImage) }
        val bestPrediction = predictions.maxByOrNull { it.score }
        Log.d("testModelWithImage", "TEST BEST PREDICTION: $bestPrediction")
        ///////////////////////////////////////////////////////////////////////////
        // Convertire il risultato preprocessato in un Bitmap
        val width = tfImage.width
        val height = tfImage.height
        val pixelData = FloatArray(width * height * 3) // RGB
        tfImage.tensorBuffer.floatArray.copyInto(pixelData)
        // Convert float RGB data to ARGB_8888 format
        val pixels = IntArray(width * height)
        for (i in pixels.indices) {
            val r = (((pixelData[i * 3] + 1) / 2) * 255).toInt().coerceIn(0, 255)
            val g = (((pixelData[i * 3 + 1] + 1) / 2) * 255).toInt().coerceIn(0, 255)
            val b = (((pixelData[i * 3 + 2] + 1) / 2) * 255).toInt().coerceIn(0, 255)
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b // ARGB
        }
        // Crea e mostra il Bitmap
        val bitmapProcessed = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmapProcessed.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmapProcessed
    }

    /** Declare and bind preview and analysis use cases */
    @SuppressLint("UnsafeExperimentalUsageError")
    private fun bindCameraUseCases() = activityCameraBinding.viewFinder.post {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            // Camera provider is now guaranteed to be available
            val cameraProvider = cameraProviderFuture.get()
            // Set up the view finder use case to display camera preview
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
                .build()
            // Set up the image analysis use case which will process frames in real time
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            //////////////////////////////////////////////////////////////////////////////////
            imageAnalysis.setAnalyzer(executor, ImageAnalysis.Analyzer { image ->
                if (!::bitmapBuffer.isInitialized) {
                    // The image rotation and RGB image buffer are initialized only once the analyzer has started running
                    imageRotationDegrees = image.imageInfo.rotationDegrees
                    bitmapBuffer =
                        Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
                }
                // Early exit: image analysis is in paused state
                if (pauseAnalysis) {
                    image.close()
                    return@Analyzer
                }
                // Copy out RGB bits to our shared buffer
                image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
                /////////////////////////////////////////////////////////////////////////////
                // Process the image in Tensorflow
                val tfImage = tfImageProcessor.process(tfImageBuffer.apply { load(bitmapBuffer) })
                // Perform the object detection for the current frame
                val predictions = synchronized(detector) { detector.predict(tfImage) }
                val bestPrediction = predictions.maxByOrNull { it.score }
                reportPrediction(bestPrediction)
            })
            //////////////////////////////////////////////////////////////////////////////////////////////////
            // Create a new camera selector each time, enforcing lens facing
            val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
            // Apply declared configs to CameraX using the same lifecycle owner
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this as LifecycleOwner,
                cameraSelector,
                preview,
                imageAnalysis
            )
            // Use the camera object to link our preview use case with the view
            preview.setSurfaceProvider(activityCameraBinding.viewFinder.surfaceProvider)
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        // Terminate all outstanding analyzing jobs (if there is any).
        executor.apply {
            shutdown()
            awaitTermination(1000, TimeUnit.MILLISECONDS)
        }
        // Release TFLite resources.
        tflite.close()
        nnApiDelegate.close()
        // Release TextToSpeech resources.
        textToSpeech.stop()
        textToSpeech.shutdown()
        super.onDestroy()
    }

    override fun onResume() {
        super.onResume()
        // Request permissions each time the app resumes, since they can be revoked at any time
        if (!hasPermissions(this)) {
            ActivityCompat.requestPermissions(
                this,
                permissions.toTypedArray(),
                permissionsRequestCode
            )
        } else {
            bindCameraUseCases()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == permissionsRequestCode && hasPermissions(this)) {
            bindCameraUseCases()
        } else {
            finish() // If we don't have the required permissions, we can't run
        }
    }

    /** Convenience method used to check if all permissions required by this app are granted */
    private fun hasPermissions(context: Context) = permissions.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }

    companion object {
        private val TAG = CameraActivity::class.java.simpleName
        private const val ACCURACY_THRESHOLD = 0.7f
        private const val MODEL_PATH = "MobileNetV2.tflite"
        private const val LABELS_PATH = "MobileNetV2_labels.txt"
        private const val TEST = false
        private const val SHOW_PROCESSED_IMAGE = true
    }
}
