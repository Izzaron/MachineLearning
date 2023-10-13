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

namespace DigitRecognizer
{
    /// <summary>
    /// Interaction logic for ImageCanvas.xaml
    /// </summary>
    public partial class ImageCanvas : UserControl
    {
        public IImageCanvasDelegate ImageCanvasDelegate;
        private readonly int imageSide = 28;
        private Rectangle[] pixels;
        public ImageCanvas()
        {
            InitializeComponent();
            InitializePixels();
        }

        private void InitializePixels()
        {
            pixels = new Rectangle[imageSide * imageSide];
            for (int i = 0; i < imageSide; i++)
            {
                for (int j = 0; j < imageSide; j++)
                {
                    Rectangle rect = new Rectangle()
                    {
                        Width = 10,
                        Height = 10,
                        Name = "pixel_" + (i * 24 + j).ToString(),
                        Fill = Brushes.White
                    };
                    pixels[i* imageSide + j] = rect;
                    rect.MouseMove += Rect_MouseOver;
                    rect.MouseDown += Rect_MouseOver;
                    canvas.Children.Add(rect);
                    Canvas.SetTop(rect, i * 10);
                    Canvas.SetLeft(rect, j * 10);
                }
            }
        }

        private void Rect_MouseOver(object sender, MouseEventArgs e)
        {
            if (sender is Rectangle pixel)
            {
                if (e.LeftButton == MouseButtonState.Pressed)
                {
                    PaintPixel(pixel);
                    ImageChanged();
                }
                else if (e.RightButton == MouseButtonState.Pressed)
                {
                    ErasePixel(pixel);
                    ImageChanged();
                }
            }
        }

        private void ImageChanged()
        {
            if (ImageCanvasDelegate != null)
                ImageCanvasDelegate.NotifyImageChanged();
        }

        private void PaintPixel(Rectangle pixel)
        {
            pixel.Fill = Brushes.Black;
            if (GetPixelNeighbour(pixel, 1, 0) is Rectangle rightNeighbour)
            {
                if (rightNeighbour.Fill != Brushes.Black)
                    rightNeighbour.Fill = Brushes.Gray;
            }
            if (GetPixelNeighbour(pixel, -1, 0) is Rectangle leftNeighbour)
            {
                if (leftNeighbour.Fill != Brushes.Black) 
                    leftNeighbour.Fill = Brushes.Gray;
            }
            if (GetPixelNeighbour(pixel, 0, 1) is Rectangle belowNeighbour)
            {
                if (belowNeighbour.Fill != Brushes.Black) 
                    belowNeighbour.Fill = Brushes.Gray;
            }
            if (GetPixelNeighbour(pixel, 0, -1) is Rectangle aboveNeighbour)
            {
                if (aboveNeighbour.Fill != Brushes.Black) 
                    aboveNeighbour.Fill = Brushes.Gray;
            }
        }

        private void ErasePixel(Rectangle pixel)
        {
            pixel.Fill = Brushes.White;
        }

        private Rectangle GetPixelNeighbour(Rectangle pixel, int stepX, int stepY)
        {
            int neighbourIndex = Array.IndexOf(pixels, pixel) + stepY * imageSide + stepX;
            if (neighbourIndex < 0 || neighbourIndex >= pixels.Length)
                return null;
            else
                return pixels[neighbourIndex];
        }

        public void clear()
        {
            foreach (Rectangle pixel in pixels)
            {
                pixel.Fill = Brushes.White;
            }
        }

        public Rectangle[] GetPixels()
        {
            return canvas.Children.OfType<Rectangle>().ToArray();
        }

        public void SetPixels(float[] newPixelValues)
        {
            for (int i = 0; i < newPixelValues.Length; i++)
            {
                byte grayScaleValue = (byte)(255.0f - newPixelValues[i]);
                var color = Color.FromRgb(grayScaleValue, grayScaleValue, grayScaleValue);
                var brushColor = new SolidColorBrush(color);
                pixels[i].Fill = brushColor;
            }
        }
    }
}
