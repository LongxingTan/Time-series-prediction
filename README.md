# Time series prediction
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)<br>
This repository implements the common methods of time series prediction, especially deep learning methods in TensorFlow2. 
It's highly welcomed to contribute if you have any better idea, just create a PR. If any question, feel free to open an issue.

#### Ongoing project, I will continue to improve this, so you might want to watch/star this repo to revisit.

<table style="width:100%" align="center">  
  <tr>
    <th>
      <p align="center">
      RNN                  
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./docs/rnn.md" name="introduction">intro</a>      
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./deepts/models/seq2seq.py" name="code">code</a>     
      </p>
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
      wavenet                 
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./docs/wavenet.md" name="introduction">intro</a>      
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./deepts/models/wavenet.py" name="code">code</a>     
      </p>
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
      transformer           
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./docs/transformer.md" name="introduction">intro</a>              
      </p>   
    </th>
    <th> 
      <p align="center">
           <a href="./deepts/models/transformer.py" name="code">code</a>     
      </p>      
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
      U-Net                  
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./docs/unet.md" name="introduction">intro</a>     
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./deepts/models/unet.py" name="code">code</a>     
      </p>      
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
      n-beats                  
      </p>
    </th>
    <th>
      <p align="center">
            <a href="./docs/nbeats.md" name="introduction">intro</a>     
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./deepts/models/nbeats.py" name="code">code</a>     
      </p>      
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
      GAN                   
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./docs/gan.md" name="introduction">intro</a>      
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./deepts/models/gan.py" name="code">code</a>     
      </p>      
    </th>
  </tr>
</table>


-----

<p align="center">
Time-series-prediction is being sponsored by the following tool; please help to support us by taking a look and signing up to a free trial
 <a href=“https://tracking.gitads.io/?repo=Time-series-prediction”> <img src="https://images.gitads.io/Time-series-prediction" alt=“GitAds”/> </a>
</p>

-----


## Usage
1. Install the library
```bash
pip install -r requirements.txt
```
2. Download the data, if necessary
```bash
bash ./data/download_passenger.sh
```
3. Train the model, set `custom_model_params` if you want, and pay attention to your own feature engineering
```bash
cd examples
python run_train.py --use_model seq2seq
```
4. Predict new data
```bash
python run_test.py
```

## Further reading
- https://github.com/awslabs/gluon-ts/
- https://github.com/Azure/DeepLearningForTimeSeriesForecasting
- https://github.com/microsoft/forecasting

## Contributor
- [LongxingTan](https://longxingtan.github.io/)

