INTRODUCTION
This project implements an AI-based smart traffic monitoring system that automatically detects traffic signal violations, rash driving behavior, and queue congestion from road surveillance videos.

The system is built from scratch using Computer Vision and Deep Learning, integrates YOLOv8-based vehicle detection & tracking, and provides a real-time Streamlit dashboard for visualization, analytics, and reporting.  

MOTIVATION
With the rapid increase in vehicles in urban areas, manual traffic monitoring is inefficient, error-prone, and non-scalable.
Traffic violations such as red-light jumping and rash driving are major contributors to road accidents.
This project aims to:
-Reduce manual monitoring
-Improve enforcement efficiency
-Provide data-driven traffic insights using AI

OBJECTIVES
-Automatically detect and track vehicles in traffic videos
-Identify red-light violations without manual signal input
-Detect rash driving based on speed estimation
-Measure queue length and queue density
-Generate replayable annotated video output
-Export detailed CSV analytics reports
-Provide an easy-to-use graphical interface

QUICK START

1️⃣ Clone Repository
git clone https://github.com/pranav-illendula/Intelligent-Traffic-Analysis---Netrik-Hackathon.git
cd INTELLIGENT-TRAFFIC-ANALYTICS

2️⃣ Run the Application
streamlit run app.py

SYSTEM OVERVIEW

System Components:
-Vehicle Detection & Tracking Module
-Violation Detection Engine
-Queue Analysis Module
-Streamlit Visualization Dashboard

Workflow:
-Upload traffic video
-Frame-by-frame vehicle detection & tracking
-Speed estimation and signal state inference
-Violation & queue analysis
-Live visualization + CSV & video output

METHODOLOGY
1️⃣Vehicle Detection & Tracking
   -Uses YOLOv8 (Ultralytics) pretrained models
   -Supports: car, bus, truck, motorcycle
   -Persistent vehicle IDs across frames

2️⃣Traffic Violation Detection:

Red-Light Jump Detection
   -Virtual stop-line placed at bottom of frame
   -Signal state inferred automatically using vehicle motion

Rash Driving Detection
  -Speed calculated using pixel displacement per second
  -Vehicles exceeding threshold flagged as rash driving

3️⃣Queue Analysis
    Queue Length: Count of stopped vehicles before stop line
    Queue Density: Ratio of occupied vehicle area to road ROI


Explanation Video:
   https://youtu.be/3kkYqmyCvxk    