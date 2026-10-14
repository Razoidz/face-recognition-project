# Face Recognition System

A Python-based face recognition application built with InsightFace, OpenCV,
and NumPy. The project detects faces in images, generates facial embeddings,
and compares them against previously enrolled faces to identify potential matches.

This project was developed as a practical exercise in Python programming,
computer vision, and machine learning concepts.

## Features

- **Face Detection:** Detect faces in images using InsightFace.
- **Face Enrollment:** Enroll individuals by associating their names with
  facial embeddings.
- **Face Recognition:** Compare detected faces against enrolled embeddings
  using Euclidean distance.
- **Threshold-Based Matching:** Determine whether a face is a potential match
  or should be classified as unknown.
- **Persistent Storage:** Save enrolled facial embeddings in an NPZ file
  using NumPy and reload them when the application restarts.
- **Interactive Menu:** Enroll people, recognize faces, view enrolled people,
  and exit through a command-line interface.
- **Input Validation:** Handle invalid image paths, empty names, duplicate
  names, and images containing zero or multiple faces during enrollment.

## Technologies Used

- Python
- InsightFace
- OpenCV
- NumPy

## How It Works

The application follows these steps:

1. Load an image using OpenCV.
2. Detect faces in the image using InsightFace.
3. Generate a numerical embedding for each detected face.
4. During enrollment, associate an embedding with a person's name.
5. Save enrolled embeddings for future use.
6. During recognition, compare a detected face's embedding against stored
   embeddings using Euclidean distance.
7. Use a distance threshold to determine whether the closest candidate
   is a potential match or the face should be classified as unknown.

### Understanding Face Embeddings

A face embedding is a numerical representation of facial features generated
by a pretrained model.

In this project, each face embedding contains 512 numerical values.
These values allow the application to compare faces mathematically rather
than comparing raw image pixels directly.

A smaller Euclidean distance indicates that two embeddings are closer in
the model's feature space. However, a small distance does not guarantee that
two images belong to the same person.

## Requirements

- Python 3.12 or another version compatible with your installed dependencies
- pip
- A working internet connection for initial dependency installation and
  any model downloads required by InsightFace
- Images containing detectable faces

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
