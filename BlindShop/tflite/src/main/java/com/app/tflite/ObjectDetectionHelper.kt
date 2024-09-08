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
    private val tflite: Interpreter,
    labelsPath: String,
    descriptionAndWeightPath: String
) {

    /** Abstraction object that wraps a prediction output in an easy to parse way */
    data class ObjectPrediction(
        val label: String,
        val score: Float,
        val weight: String,
        val description: String
    )

    private val labels: List<String> = loadLabels(labelsPath)
    private val weights: List<String> = loadWeights(descriptionAndWeightPath)
    private val descriptions: List<String> = loadDescription(descriptionAndWeightPath)

    private val outputBuffer = arrayOf(FloatArray(labels.size))

    @Synchronized
    fun predict(image: TensorImage): List<ObjectPrediction> {
        return try {
            tflite.run(image.buffer, outputBuffer)

            val predictions = outputBuffer[0].mapIndexed { index, score ->
                ObjectPrediction(
                    label = labels[index],
                    score = score,
                    weight = weights[index],
                    description = descriptions[index]
                )
            }

            val sortedPredictions = predictions.sortedByDescending { it.score }
            val predictionsString = sortedPredictions.joinToString(separator = "\n") {
                "Label = ${it.label}, Score = ${it.score}, Peso = ${it.weight}, Desc = ${it.description}"
            }
            Log.d("ObjectDetectionHelper", "Sorted predictions:\n$predictionsString")

            return sortedPredictions
        } catch (e: Exception) {
            Log.e("ObjectDetectionHelper", "Error during prediction: ", e)
            emptyList() // Return an empty list in case of an error
        }
    }

    private fun loadLabels(path: String): List<String> {
        val labels = mutableListOf<String>()
        try {
            val inputStream = context.assets.open(path)
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

    private fun loadWeights(path: String): List<String> {
        val weights = mutableListOf<String>()
        try {
            val inputStream = context.assets.open(path)
            BufferedReader(InputStreamReader(inputStream)).use { reader ->
                reader.forEachLine { line ->
                    weights.add(line.split('/')[1])
                }
            }
        } catch (e: Exception) {
            Log.e("ObjectDetectionHelper", "Error loading weights: ", e)
        }
        return weights
    }

    private fun loadDescription(path: String): List<String> {
        val descriptions = mutableListOf<String>()
        try {
            val inputStream = context.assets.open(path)
            BufferedReader(InputStreamReader(inputStream)).use { reader ->
                reader.forEachLine { line ->
                    descriptions.add(line.split('/')[2])
                }
            }
        } catch (e: Exception) {
            Log.e("ObjectDetectionHelper", "Error loading descriptions: ", e)
        }
        return descriptions
    }
}
