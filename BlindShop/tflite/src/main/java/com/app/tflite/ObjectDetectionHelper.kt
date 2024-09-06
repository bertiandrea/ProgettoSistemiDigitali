package com.app.tflite

import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage

/**
 * Helper class used to communicate between our app and the TF object detection model
 */
class ObjectDetectionHelper(private val tflite: Interpreter, private val labels: List<String>) {

    /** Abstraction object that wraps a prediction output in an easy to parse way */
    data class ObjectPrediction(val label: String, val score: Float)

    private val outputBuffer = arrayOf(FloatArray(OBJECT_COUNT))

    @Synchronized
    fun predict(image: TensorImage): List<ObjectPrediction> {
        return try {
            tflite.run(image.buffer, outputBuffer)

            val predictions = outputBuffer[0].mapIndexed { index, score ->
                ObjectPrediction(
                    label = labels[index],
                    score = score
                )
            }

            val sortedPredictions = predictions.sortedByDescending { it.score }
            val predictionsString = sortedPredictions.joinToString(separator = "\n") {
                "Label = ${it.label}, Score = ${it.score}"
            }
            Log.d("ObjectDetectionHelper", "Sorted predictions:\n$predictionsString")

            return predictions
        } catch (e: Exception) {
            Log.e("ObjectDetectionHelper", "Error during prediction: ", e)
            emptyList() // Return an empty list in case of an error
        }
    }

    companion object {
        const val OBJECT_COUNT = 37 //Numero Classi
    }
}