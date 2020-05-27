# Time series prediction
This repo implements the common methods of time series prediction, especially deep learning methods in TensorFlow2. 
It's highly welcomed to contribute if you have better idea, just create a PR. If any question, feel free to open an issue.


<table style="width:100%">
  <tr>
    <th>
      <p align="center">
      ARIMA                      
      </p>
    </th>
    <th>
      <p align="center">
           <a href="./docs/arima.md" name="introduction">intro</a>             
      </p>
       <p align="center">           
           <a href="./deepts/models/arima.py" name="code">code</a>     
      </p>
    </th> 
  </tr>
  <tr>
    <th>
      <p align="center">
      GBDT                  
      </p>
    </th>
    <th>
      <p align="center">
        <a href="./docs/tree.md" name="introduction">intro</a> 
      </p>   
      <p align="center">
           <a href="./deepts/models/tree.py" name="code">code</a>     
      </p>
    </th>
  </tr>
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
      <p align="center">
           <a href="./deepts/models/gan.py" name="code">code</a>     
      </p>      
    </th>
  </tr>
</table>


## Usage
1. Install the library
```bash
pip install -r requirements.txt
```
2. Download the data, if necessary
```bash
bash ./data/download_passenger.sh
```
3. Train the model, set `custom_model_params` if you want
```bash
cd examples
python run_train.py --use_model seq2seq
```
4. Predict new data
```bash
python run_test.py
```

## Further reading
https://github.com/awslabs/gluon-ts/

## Contributor
- [LongxingTan](https://longxingtan.github.io/)
