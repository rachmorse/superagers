# fMRI Analysis

Brief description of project.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Provide a more detailed description of your project, its purpose, and its features.

# Key Terms

Framewise Displacement (FWD):
     A measure in fMRI analysis to quantify the amount of head movement between consecutive scans. It captures both translational (linear) and rotational (angular) movements, providing a single value that indicates the total displacement of the head. FWD is calculated as the sum of the absolute differences in head position and rotation between successive frames.

Interpolation: 
    The process of estimating unknown values (in this case value we removed due to high FWD values) that are within the range of known values

    Example:
        Imagine you have data points for tp 1, 2, 4, and 5, but the value for tp 3 has been removed because of high movement at tp 3:

        Time:   1   2  3   4   5
        Value: 10  20  ??  40  50

        Interpolation estimates the missing value at tp 3 based on known neighboring values. 

Extrapolation:
    The process of estimating unknown values (in this case value we removed due to high FWD values) that are outside of the range of known values

    Example:
        Imagine you have data points at tp 1, 2, 3, and 4, but you need to estimate the value at tp 5:

        Time:   1   2   3   4  5
        Value: 10  20  30  40 ??

        Using extrapolation, you estimate the value at tp 5 based on the trend from the known values. 

## Installation

Instructions on how to install and set up your project. For example:

### Python

```sh
git clone https://github.com/yourusername/yourproject.git
cd yourproject
pip install -r requirements.txt