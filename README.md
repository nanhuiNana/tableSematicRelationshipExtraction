# Table Sematic Relationship Extraction
* file struct
  - baseline
    - train.py # baseline train code(paddle version)
    - infer.py # baseline infer code(paddle version)
  - dataset
    - train.zip # train dataset
      - Train_Set
        - xxx.csv
    - test.csv # test dataset
    - labels.csv # labels set
* environment
  * cuda 11.8 version install command:
    ```
    python -m pip install --pre paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    ```
* run command
  ```
  python train.py
  python infer.py
  ```
