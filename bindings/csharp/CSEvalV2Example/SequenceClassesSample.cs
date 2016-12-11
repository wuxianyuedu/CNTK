using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;
using Microsoft.MSR.CNTK.Extensibility.Managed.CSEvalClient;

namespace CSEvalV2Example
{
    public class SequenceClassesSample
    {
        // The example shows how to use Sequence class.
        static void DenseExample()
        {
            const string outputNodeName = "Plus2060_output";

            // Load the model.
            Function modelFunc = Function.LoadModel("z.model");

            // Todo: how to get a variable in the intermeidate layer by name?
            Variable outputVar = modelFunc.Outputs.Where(variable => string.Equals(variable.Name, outputNodeName)).Single();

            // Set desired output variables and get required inputVariables;
            Function evalFunc = Function.Combine(new Variable[] { outputVar });

            // Only signle input variable
            Variable inputVar = evalFunc.Arguments.Single();

            // Get shape data for the input variable
            NDShape inputShape = inputVar.Shape;
            // Todo: add property to Shape
            uint imageWidth = inputShape[0];
            uint imageHeight = inputShape[1];
            uint imageChannels = inputShape[2];
            uint imageSize = inputShape.TotalSize;

            // Number of sequences for this batch
            int numOfSequences = 2;
            // Number of samples in each sequence
            int[] numOfSamplesInSequence = { 3, 3 };

            // inputData contains mutliple sequences. Each sequence has multiple samples.
            // Each sample has the same tensor shape.
            // The outer List is the sequences. Its size is numOfSequences.
            // The inner List is the inputs for one sequence. Its size is inputShape.TotalSize * numberOfSampelsInSequence
            var inputBatch = new List<Sequence<float>>();
            var fileList = new List<string>() { "00000.png", "00001.png", "00002.png", "00003.png", "00004.png", "00005.png" };
            int fileIndex = 0;
            for (int seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                // Create a new data buffer for the new sequence
                var seqData = new Sequence<float>(inputVar.Shape);
                for (int sampleIndex = 0; sampleIndex < numOfSamplesInSequence[seqIndex]; sampleIndex++)
                {
                    Bitmap bmp = new Bitmap(Bitmap.FromFile(fileList[fileIndex++]));
                    var resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                    List<float> resizedCHW = resized.ParallelExtractCHW();
                    // Aadd this sample to the data buffer of this sequence
                    seqData.AddRange(resizedCHW);
                }
                // Add this sequence to the sequences list
                inputBatch.Add(seqData);
            }

            // Create input map
            var inputMap = new Dictionary<Variable, Value>();
            // void Create<T>(Shape shape, List<List<T>> data, DeviceDescriptor computeDevice)
            inputMap.Add(inputVar, Value.Create(inputBatch, DeviceDescriptor.CPUDevice));

            // Create ouput map. Using null as Value to indicate using system allocated memory.
            var outputMap = new Dictionary<Variable, Value>();
            outputMap.Add(outputVar, null);

            // Evalaute
            // Todo: test on GPUDevice()?
            // Use Variables in input and output maps.
            // It is also possible to use variable name in input and output maps.
            evalFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            // The buffer for storing output for this batch
            var outputData = new List<Sequence<float>>();
            Value outputVal = outputMap[outputVar];
            // Get output result as dense output
            // void CopyTo(List<List<T>>
            outputVal.CopyTo(outputVar, outputData);

            // Output results
            var numOfElementsInSample = outputVar.Shape.TotalSize;
            uint seqNo = 0;
            foreach (var seq in outputData)
            {
                uint elementIndex = 0;
                uint sampleIndex = 0;
                foreach (var data in seq)
                {
                    // a new sample starts.
                    if (elementIndex++ == 0)
                    {
                        Console.Write("Seq=" + seqNo + ", Sample=" + sampleIndex + ":");
                    }
                    Console.Write(" " + data);
                    // reach the end of a sample.
                    if (elementIndex == numOfElementsInSample)
                    {
                        Console.WriteLine(".");
                        elementIndex = 0;
                        sampleIndex++;
                    }
                }
                seqNo++;
            }
        }
    }
}
