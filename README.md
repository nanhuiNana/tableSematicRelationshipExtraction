# Table Sematic Relationship Extraction
* file struct
  - baseline
    - train.py
    - infer.py
  - dataset
    - train.zip
      - Train_Set
        - xxx.csv    
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
