# 📘 **API Documentation for ML Prediction Service**

This document provides a comprehensive guide to using the ML Prediction Service built with **FastAPI**. This service hosts multiple machine learning models for predictions using **Random Forest**, **Decision Tree**, and **Deep Neural Networks (DNN)**. Users can make predictions for pre-training and post-training datasets. 

---

## **🛠️ Base URL**
```
http://localhost:8000/
```
> Replace `localhost:8000` with the URL where the service is deployed.

---

## **📋 Table of Contents**
1. [Introduction](#-introduction)
2. [API Endpoints](#-api-endpoints)
    - [GET /](#1-get-)
    - [POST /predict/dnn](#2-post-predictdnn)
    - [POST /predict/random-forest](#3-post-predictrandom-forest)
    - [POST /predict/decision-tree](#4-post-predictdecision-tree)
    - [POST /predict/dnn-post](#5-post-predictdnn-post)
    - [POST /predict/random-forest-post](#6-post-predictrandom-forest-post)
    - [POST /predict/decision-tree-post](#7-post-predictdecision-tree-post)
3. [Request Body Schema](#-request-body-schema)
4. [Response Structure](#-response-structure)
5. [Error Handling](#-error-handling)
6. [Example Requests](#-example-requests)
7. [Common Issues and Solutions](#-common-issues-and-solutions)

---

## 🔥 **Introduction**
This API service allows users to interact with pre-trained machine learning models and get predictions for various tasks. It exposes the following types of models:
- **Random Forest** (pre/post)
- **Decision Tree** (pre/post)
- **Deep Neural Networks (DNN)** (pre/post)

> The models are pre-loaded from `.keras`, `.pkl`, and other model files at runtime.

---

## 📌 **API Endpoints**

### 1️⃣ **GET /**
> **Description:** Root endpoint that displays an introductory message about the available endpoints.

**Request**  
```
GET / 
```

**Response**
```json
{
    "message": "Welcome to the ML API! 1) Use /predict/random-forest and /predict/random-forest-post 2) Use /predict/decision-tree and /predict/decision-tree-post 3) Use /predict/dnn and /predict/dnn-post to make predictions."
}
```

---

### 2️⃣ **POST /predict/dnn**
> **Description:** Predicts using the **Deep Neural Network (DNN)** on the pre-trained dataset.

**Request**  
```
POST /predict/dnn
```

**Request Body**
```json
{
    "feature_vector": [value1, value2, value3, ..., valueN]
}
```

**Response**
```json
{
    "prediction": 1
}
```

**Errors**
- **422**: Validation error (when the feature vector is invalid)
- **500**: Internal server error

---

### 3️⃣ **POST /predict/random-forest**
> **Description:** Predicts using the **Random Forest** model on the pre-trained dataset.

**Request**  
```
POST /predict/random-forest
```

**Request Body**
```json
{
    "feature_vector": [value1, value2, value3, ..., valueN]
}
```

**Response**
```json
{
    "prediction": 0
}
```

**Errors**
- **500**: Internal server error

---

### 4️⃣ **POST /predict/decision-tree**
> **Description:** Predicts using the **Decision Tree** model on the pre-trained dataset.

**Request**  
```
POST /predict/decision-tree
```

**Request Body**
```json
{
    "feature_vector": [value1, value2, value3, ..., valueN]
}
```

**Response**
```json
{
    "prediction": 1
}
```

**Errors**
- **500**: Internal server error

---

### 5️⃣ **POST /predict/dnn-post**
> **Description:** Predicts using the **Deep Neural Network (DNN)** on the post-trained dataset.

**Request**  
```
POST /predict/dnn-post
```

**Request Body**
```json
{
    "feature_vector": [value1, value2, value3, ..., valueN]
}
```

**Response**
```json
{
    "prediction": 0
}
```

**Errors**
- **422**: Validation error (when the feature vector is invalid)
- **500**: Internal server error

---

### 6️⃣ **POST /predict/random-forest-post**
> **Description:** Predicts using the **Random Forest** model on the post-trained dataset.

**Request**  
```
POST /predict/random-forest-post
```

**Request Body**
```json
{
    "feature_vector": [value1, value2, value3, ..., valueN]
}
```

**Response**
```json
{
    "prediction": 1
}
```

**Errors**
- **500**: Internal server error

---

### 7️⃣ **POST /predict/decision-tree-post**
> **Description:** Predicts using the **Decision Tree** model on the post-trained dataset.

**Request**  
```
POST /predict/decision-tree-post
```

**Request Body**
```json
{
    "feature_vector": [value1, value2, value3, ..., valueN]
}
```

**Response**
```json
{
    "prediction": 0
}
```

**Errors**
- **500**: Internal server error

---

## 📚 **Request Body Schema**
All POST endpoints follow the same request schema.

| **Field**         | **Type**      | **Required** | **Description**            |
|-------------------|---------------|--------------|----------------------------|
| **feature_vector** | List of floats | ✅ Required  | Array of features for the model |

---

## 📘 **Response Structure**
Each prediction endpoint returns the following JSON response structure.

| **Field**       | **Type**       | **Description**           |
|-----------------|----------------|----------------------------|
| **prediction**  | int (0 or 1)    | The predicted class (0 or 1) |

---

## ⚠️ **Error Handling**
If an error occurs, the API returns a proper error message with a relevant status code.  

---

## 🔥 **Example Requests**
### 📡 **Prediction Using DNN**
```json
{
    "feature_vector": [1.2, 0.8, 3.4, 2.1, 0.6, 0.7, 1.0, 2.5, 0.9, 1.1, 0.4, 0.6, 1.8, 0.3, 2.2, 3.3]
}
```

---

## 🛠️ **Common Issues and Solutions**
| **Issue**                    | **Cause**                   | **Solution**                      |
|------------------------------|-----------------------------|-----------------------------------|
| Missing **feature_vector**    | Request body is empty       | Ensure you send the body correctly|
| Model file not found          | Model paths are incorrect   | Check file paths in `BASE_DIR`    |
| Internal Server Error         | Data shape mismatch         | Ensure feature vector has the correct size |
| Validation Error (422)        | Invalid input data          | Provide a valid array of floats    |
