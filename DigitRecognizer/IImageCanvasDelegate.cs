using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitRecognizer
{
    public interface IImageCanvasDelegate
    {
        void NotifyImageChanged();
    }
}
