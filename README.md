# NoiseDFD

### Usage

To run the method, simply provide the path to the input document image. The script will process the image and save several outputs:

- `characters.png`: Visualization of the extracted characters (connected components).
- `stds.png`: Shows each detected character with a value representing its standard deviation.
- `outliers.png`: Highlights characters whose standard deviations fall outside the fluctuation interval defined by significance level `a`.
- `words.png`: Displays the words extracted using the EAST text detector.
- `nfa.png`: Final word-level detection based on a Number of False Alarms (NFA) threshold `t`.

All output images are saved in the current working directory.


An example on how to run the method is given below:
```
python main.py input_document.png -b 10 -a 0.1 -t 0.1
```

Where the arguments are:
- `-b`:	Blobs removal threshold: connected components with area smaller than this value will be discarded.
- `-a`:	Significance level used for detecting outliers based on standard deviation.
- `-t`: NFA threshold used for final word-level detection.

### Online demo

You can try the method online in the following <a href="https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000542">IPOL demo</a>.
