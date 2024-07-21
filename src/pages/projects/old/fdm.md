---
layout: ../../layouts/ProjectLayout.astro
title: 'Flight Dynamic Model'
cover: /images/fdm.png
description: 'Flight Dynamic Model based on LSTM and Dense Layers made with Tensorflow and Keras. The model was trained with data from real Quadplane drone flights ans is able to predict the next state of the drone based on the previous states. Along with the model, a simple Desktop App was made to train and validate the model.'
---

## 1. Backend

### 1.1. Technologies
- Python
- Tensorflow and Keras
- FastAPI


### 1.2. mavlogdump script
`mavlogdump.py` is a fork of a script in [pymavlink Github repo](https://github.com/ArduPilot/pymavlink/blob/7b0d51cca7e75b3cf84f5dbb74e76f727816e50d/tools/mavlogdump.py) adding the functionality of being able to save the extracted data in the supported formats as well as supporting its use from other modules.

#### Args options

| Option String          | Required | Choices                     | Default         | Option Summary                                               |
| ---------------------- | -------- | --------------------------- | --------------- | ------------------------------------------------------------ |
| ["--h", "--help"]      | False    |                             |                 | Show this help message and exit                              |
| ["--no-timestamps"]    | False    |                             |                 | Log doesn't have timestamps                                  |
| ["--planner"]          | False    |                             |                 | Use planner file format                                      |
| ["--robust"]           | False    |                             |                 | Enable robust parsing (skip over bad data)                   |
| ["-f", "--follow"]     | False    |                             |                 | Keep waiting for more data at end of file                    |
| ["--condition"]        | False    |                             | None            | Select packets by condition                                  |
| ["-q", "--quiet"]      | False    |                             |                 | Don't display packets                                        |
| ["-o", "--output"]     | False    |                             | None            | Output matching packets to give file                         |
| ["-p", "--parms"]      | False    |                             |                 | Preserve parameters in output with -o                        |
| ["--format"]           | False    | ["standard", "json", "csv"] | None            | Change the output format between 'standard', 'json', and 'csv'. For the CSV output, you must supply types that you want. |
| ["--csv_sep"]          | False    |                             |                 | Select the delimiter between columns for the output CSV file. Use 'tab' to specify tabs. Only applies when --format=csv |
| ["--types"]            | False    |                             | None            | Types of messages (comma separated with wildcard)            |
| ["--nottypes"]         | False    |                             | None            | Types of messages not to include (comma separated with wildcard) |
| ["--dialect"]          | False    |                             | "ardupilotmega" | MAVLink dialect                                              |
| ["--zero-time-base"]   | False    |                             |                 | Use Z time base for DF logs                                  |
| ["--no-bad-data"]      | False    |                             |                 | Don't output corrupted messages                              |
| ["--show-source"]      | False    |                             |                 | Show source system ID and component ID                       |
| ["--show-seq"]         | False    |                             |                 | Show sequence numbers                                        |
| ["--source-system"]    | False    |                             | None            | Filter by source system ID                                   |
| ["--source-component"] | False    |                             | None            | Filter by source component ID                                |
| ["--link"]             | False    |                             |                 | Filter by comms link ID                                      |
| ["--save"]             | False    |                             |                 | Save output to a file                                        |
| ["log"]                | True     |                             |                 | File to read                                                 |

### 1.3. DataFlashLogs Module
DataFlash Logs acts as the front end of the script (facade pattern) and parses the extracted data so that it can be easily manipulated by other tools. In addition, it allows exporting and importing these data to an excel file, selecting fields of interest from all messages and export it to a single csv, as well as plotting the desired fields.

### 1.4. FlightDynamicModel Module
Contains the FlightDynamicsModel class which instance a DataFlashLogs object and define methods for parsing and preparing data for training. It also implement methods to create the keras model, training and testing as well as plotting history. All parameters for datasets, model and training are set as attributes.

## 2. Frontend

- PySide6


## 3. Development

### 3.1. Backend

#### Environment
```bash
conda env create -f environment.yml
conda activate fdm
```

#### Docker
```bash
mkdir ~/mariadb
sudo chmod -R 777 ~/mariadb
sudo apt install graphviz # for pydot (keras plot_model)
pip-compile requirements.in # generate requirements.txt
docker pull mariadb
docker pull nginx
docker build -f Dockerfile.api -t fdmapi .
docker build -f Dockerfile.docs -t fdmdocs .
docker compose --env-file .env up -d 
```