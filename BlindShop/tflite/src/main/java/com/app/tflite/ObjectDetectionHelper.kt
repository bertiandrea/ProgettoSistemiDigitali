package com.app.tflite

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Helper class used to communicate between our app and the TF object detection model
 */
class ObjectDetectionHelper(
    private val context: Context,
    private val tflite: Interpreter
) {

    /** Abstraction object that wraps a prediction output in an easy to parse way */
    data class ObjectPrediction(val label: String, val score: Float)

    private val labels: List<String> = loadLabels()
    private val outputBuffer = arrayOf(FloatArray(labels.size))

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

            sortedPredictions
        } catch (e: Exception) {
            Log.e("ObjectDetectionHelper", "Error during prediction: ", e)
            emptyList() // Return an empty list in case of an error
        }
    }

    private fun loadLabels(): List<String> {
        val labels = mutableListOf<String>()
        try {
            val inputStream = context.assets.open("MobileNetV2_labels.txt")
            BufferedReader(InputStreamReader(inputStream)).use { reader ->
                reader.forEachLine { line ->
                    labels.add(line)
                }
            }
        } catch (e: Exception) {
            Log.e("ObjectDetectionHelper", "Error loading labels: ", e)
        }
        return labels
    }
}
