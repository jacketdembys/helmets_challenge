# Import the required libraries
import datetime
import shutil
from pathlib import Path
from collections import Counter
import yaml
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.model_selection import KFold