using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Microsoft.Win32;

namespace DigitRecognizer
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window, IImageCanvasDelegate
    {
        private NeuralNetwork neuralNetwork;
        private string imagePath;
        private string labelPath;
        public MainWindow()
        {
            InitializeComponent();

            this.KeyDown += MainWindow_KeyDown;

            imageCanvas.ImageCanvasDelegate = this;

            //int[] layerLengths = { 28 * 28, 512, 512, 10 };
            int[] layerLengths = { 28 * 28, 64, 64, 10 };
            neuralNetwork = new NeuralNetwork(layerLengths);
            outputTextBox.Text = "No weights loaded.";
        }

        private void MainWindow_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.R)
            {
                imageCanvas.clear();
                UpdateGuess();
            }
        }

        public void NotifyImageChanged()
        {
            UpdateGuess();
        }

        private void UpdateGuess()
        {
            Rectangle[] pixels = imageCanvas.GetPixels();

            Color c = (pixels[0].Fill as SolidColorBrush).Color;

            float[] neuralInput = pixels.Select(pixel => 255.0f - (pixel.Fill as SolidColorBrush).Color.B).ToArray();

            float[] neuralOutput = neuralNetwork.MakePrediction(neuralInput);

            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < neuralOutput.Length; i++)
            {
                if (neuralOutput[i] * 100 < 50)
                    continue;
                sb.Append(i);
                sb.Append(": ");
                sb.Append((neuralOutput[i] * 100).ToString("n1"));
                sb.AppendLine(" %");
            }

            outputTextBox.Text = sb.ToString();
        }

        private void trainButton_Click(object sender, RoutedEventArgs e)
        {
            neuralNetwork.NNAddWithCuda();
            /*if (imagePath == null)
                imagePath = getFilePathWithDialog("Images file|train-images.idx3-ubyte");

            if (labelPath == null)
                labelPath = getFilePathWithDialog("Labels file|train-labels.idx1-ubyte");

            if (imagePath != null && labelPath != null)
            {
                int nrOfEpochs = Int32.Parse(nrOfEpochsTextBox.Text);
                outputTextBox.Text = "Training started";
                if (neuralNetwork.TrainOnMnist(imagePath, labelPath, nrOfEpochs))
                {
                    outputTextBox.Text = "Successfully trained for " + nrOfEpochs.ToString() + " epochs.";
                }
            }*/
        }

        private string getFilePathWithDialog(string filter)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.InitialDirectory = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            openFileDialog.Filter = filter;
            if (openFileDialog.ShowDialog() == true)
                return openFileDialog.FileName;
            return null;
        }

        private void loadButton_Click(object sender, RoutedEventArgs e)
        {
            if (neuralNetwork.LoadWeights())
            {
                outputTextBox.Text = "Weights loaded successfully!";
            }
            else
            {
                outputTextBox.Text = "Weights could not be loaded.";
            }
        }
    }
}
