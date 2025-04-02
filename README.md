# NoiseDFD

### Usage
One must provide the path to the document image and to the corresponding words mask. The default outputs are `characters.png` which shows the extracted characters (connected components), `stds.png` where each of the detected characters is assigned a value related to its standard deviation, `outliers.png` where those characters with stds that fall outside the fluctuation interval at level `a` are shown and, finally, `nfa.png` which is the final detection made at a word level with a number of false alarms threshold set to `t`. All outputs are stored in the current directory.   

An example on how to run the method is given below:
```
python main.py input_document.png input_mask.png -a 0.1 -t 0.1
```
