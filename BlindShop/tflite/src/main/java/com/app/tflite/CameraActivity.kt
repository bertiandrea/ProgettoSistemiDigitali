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
import android.view.MotionEvent
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.GestureDetectorCompat
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
    private lateinit var gestureDetector: GestureDetectorCompat

    private val executor = Executors.newSingleThreadExecutor()
    private val permissions = listOf(Manifest.permission.CAMERA)
    private val permissionsRequestCode = Random.nextInt(0, 10000)

    private var pauseAnalysis = false
    private var imageRotationDegrees: Int = 0
    private val tfImageBuffer =
        TensorImage(DataType.FLOAT32) //DataType.FLOAT32 or DataType.UINT8 depending on MODEL

    private var bestPrediction: ObjectDetectionHelper.ObjectPrediction? = null

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
        ObjectDetectionHelper(this, tflite, LABELS_PATH, DESCRIPTION_WEIGHT_PATH)
    }
    private val tfInputSize by lazy {
        val inputIndex = 0
        val inputShape = tflite.getInputTensor(inputIndex).shape()
        Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
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
                if(CONSTANT_REPORT) {
                    // Process the image in Tensorflow
                    val tfImage = tfImageProcessor.process(tfImageBuffer.apply { load(bitmapBuffer) })
                    // Perform the object detection for the current frame
                    val predictions = synchronized(detector) { detector.predict(tfImage) }
                    val bestPrediction = predictions.maxByOrNull { it.score }
                    reportPrediction(bestPrediction)
                }
            })
            //////////////////////////////////////////////////////////////////////////////////////////////////
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this as LifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalysis
            )
            // Use the camera object to link our preview use case with the view
            preview.setSurfaceProvider(activityCameraBinding.viewFinder.surfaceProvider)
        }, ContextCompat.getMainExecutor(this))
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityCameraBinding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(activityCameraBinding.root)

        initializeTextToSpeech()
        initializeGestureDetector()
        initializeTouchListener()
    }

    private fun initializeTextToSpeech() {
        textToSpeech = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.language = Locale.ITALIAN
            } else {
                Log.e(TAG, "TextToSpeech initialization failed")
            }
        }
    }

    private fun initializeGestureDetector() {
        val swipeListener = object : SwipeGestureListener() {
            override fun onSwipeLeft() {
                handleSwipeLeft()
            }
            override fun onSwipeRight() {
                handleSwipeRight()
            }
        }
        gestureDetector = GestureDetectorCompat(this, swipeListener)
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun initializeTouchListener() {
        activityCameraBinding.fullscreenTouchCapture.setOnTouchListener { _, event ->
            gestureDetector.onTouchEvent(event)
        }
        activityCameraBinding.fullscreenTouchCapture.setOnClickListener {
            handleTouchClick(it)
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

    private fun rotateAndMirrorBitmap(bitmap: Bitmap, rotationDegrees: Int, isFrontFacing: Boolean): Bitmap {
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
            activityCameraBinding.textPrediction.text = prediction.label.uppercase()
            //activityCameraBinding.textPrediction.text =
            //    "${"%.0f".format(prediction.score * 100)}%\n${prediction.label}"
        }

    private fun speakOut(prediction: ObjectDetectionHelper.ObjectPrediction?) {
        if (prediction == null || prediction.score < ACCURACY_THRESHOLD) {
            textToSpeech.speak("Sconosciuto", TextToSpeech.QUEUE_FLUSH, null, null)
        } else {
            textToSpeech.speak(prediction.label, TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }

    private fun testModelWithImage(): Bitmap {
        val inputStream = assets.open("testImage.jpg")
        val bitmapBuffer = BitmapFactory.decodeStream(inputStream)

        // Esegui il resize dell'immagine
        val targetWidth = 224  // Imposta la larghezza desiderata
        val targetHeight = 224 // Imposta l'altezza desiderata
        val resizedBitmap = Bitmap.createScaledBitmap(bitmapBuffer, targetWidth, targetHeight, true)

        // Crea una matrice per la rotazione di 90 gradi
        val matrix = Matrix()
        matrix.postRotate(-90f)  // Ruota di 90 gradi in senso antiorario

        // Applica la rotazione alla bitmap ridimensionata
        val rotatedBitmap = Bitmap.createBitmap(resizedBitmap, 0, 0, resizedBitmap.width, resizedBitmap.height, matrix, true)

        // Pre-elabora l'immagine ruotata
        val tfImage = tfImageProcessor.process(tfImageBuffer.apply { load(rotatedBitmap) })
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

    private fun handleSwipeLeft() {
        if (pauseAnalysis) {
            Log.e(TAG, "handleSwipeLeft")
            if (bestPrediction == null || bestPrediction!!.score < ACCURACY_THRESHOLD || bestPrediction!!.weight == "") {
                textToSpeech.speak("Peso non disponibile", TextToSpeech.QUEUE_FLUSH, null, null)
            } else {
                textToSpeech.speak(bestPrediction!!.weight , TextToSpeech.QUEUE_FLUSH, null, null)
            }
        }
    }

    private fun handleSwipeRight() {
        if (pauseAnalysis) {
            Log.e(TAG, "handleSwipeRight")
            if (bestPrediction == null || bestPrediction!!.score < ACCURACY_THRESHOLD || bestPrediction!!.description == "") {
                textToSpeech.speak("Descrizione non disponibile", TextToSpeech.QUEUE_FLUSH, null, null)
            } else {
                textToSpeech.speak(bestPrediction!!.description , TextToSpeech.QUEUE_FLUSH, null, null)
            }
        }
    }

    private fun handleTouchClick(view: View) {
        Log.e(TAG, "handleTouchClick")
        view.isEnabled = false
        if (pauseAnalysis) {
            textToSpeech.speak("Analizzo", TextToSpeech.QUEUE_FLUSH, null, null)
            pauseAnalysis = false
            activityCameraBinding.imagePredicted.visibility = View.GONE
            if (!CONSTANT_REPORT) {
                activityCameraBinding.textPrediction.visibility = View.GONE
            }
        } else {
            if (TEST) {
                val bitmapImage = testModelWithImage()
                activityCameraBinding.imagePredicted.setImageBitmap(bitmapImage)
                val tfImage = tfImageProcessor.process(tfImageBuffer.apply { load(bitmapImage) })
                val predictions = synchronized(detector) { detector.predict(tfImage) }
                bestPrediction = predictions.maxByOrNull { it.score }
                pauseAnalysis = true
                reportPrediction(bestPrediction)
                speakOut(bestPrediction)
            } else {
                val tfImage = tfImageProcessor.process(tfImageBuffer.apply { load(bitmapBuffer) })
                val predictions = synchronized(detector) { detector.predict(tfImage) }
                bestPrediction = predictions.maxByOrNull { it.score }
                pauseAnalysis = true
                reportPrediction(bestPrediction)
                speakOut(bestPrediction)
                val bitmap = if (SHOW_PROCESSED_IMAGE) {
                    convertTfImageToBitmap(tfImage)
                } else {
                    rotateAndMirrorBitmap(bitmapBuffer, imageRotationDegrees, false)
                }
                activityCameraBinding.imagePredicted.setImageBitmap(bitmap)
            }
            activityCameraBinding.imagePredicted.visibility = View.VISIBLE
        }
        view.isEnabled = true
    }

    override fun onDestroy() {
        super.onDestroy()
        // Terminate all outstanding analyzing jobs (if there is any).
        executor.shutdown()
        try {
            if (!executor.awaitTermination(1, TimeUnit.SECONDS)) {
                executor.shutdownNow()
            }
        } catch (e: InterruptedException) {
            executor.shutdownNow()
        }
        // Release TFLite resources.
        tflite.close()
        nnApiDelegate.close()
        // Release TextToSpeech resources.
        textToSpeech.stop()
        textToSpeech.shutdown()
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

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == permissionsRequestCode && hasPermissions(this)) {
            bindCameraUseCases()
        } else {
            finish() // If we don't have the required permissions, we can't run
        }
    }

    private fun hasPermissions(context: Context) = permissions.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }

    companion object {
        private val TAG = CameraActivity::class.java.simpleName
        private const val ACCURACY_THRESHOLD = 0.50f
        private const val MODEL_PATH = "MobileNetV2.tflite"
        private const val LABELS_PATH = "MobileNetV2_labels.txt"
        private const val DESCRIPTION_WEIGHT_PATH = "MobileNetV2_descriptions.txt"
        private const val TEST = false
        private const val SHOW_PROCESSED_IMAGE = false
        private const val CONSTANT_REPORT = false
    }
}
