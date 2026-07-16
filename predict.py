#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Author: wellybreeze xuekai9595@qq.com
Date: 2026-07-16 01:14:04
LastEditors: wellybreeze xuekai9595@qq.com
LastEditTime: 2026-07-16 01:14:08
FilePath: /predict.py
Description: 

Copyright (c) 2026 by wellybreeze (xuekai9595@qq.com), All Rights Reserved. 
'''
from ultralytics import RFDETR
model = RFDETR("pretrained_weights/detect/rf-detr-nano.pth")
model.predict("ultralytics/assets/bus.jpg", save=True)
