<!--
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <!-- <a href="https://github.com/dastratakos/Face-Mask-Detection">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <h1 align="center">Face Mask Detection</h1>

  <p align="center">
    A machine learning model for a <a href="http://cs229.stanford.edu">CS 229</a> final project
    <br />
    <a href="https://github.com/dastratakos/Face-Mask-Detection"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/dastratakos/Face-Mask-Detection">View Demo</a>
    ·
    <a href="https://github.com/dastratakos/Face-Mask-Detection/issues">Report Bug</a>
    ·
    <a href="https://github.com/dastratakos/Face-Mask-Detection/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

<!-- ABOUT THE PROJECT -->
## About The Project

Across the globe, there have been 33.8 million reported coronavirus cases.
Covid-19 has plunged countless nations into chaos and recession as they scramble
to keep the virus contained. Due to the highly contagious nature of the virus,
every individual must do their part in preventing the spread by taking
precautions such as wearing a face mask. Yet there are still many individuals
who refuse to do so. This careless behavior puts many lives at risk, so it is
imperative that we hold individuals responsible for protecting the general
public.

In light of this issue, our project aims to create a machine learning model that
can accurately detect, given an image, whether a person is properly wearing a
face mask or not. This project will especially be important in the global return
to work effort as businesses continue to search for ways to keep their employees
and customers safe. Automating the process of face mask detection will reduce
human labor while creating a system of accountability.

With regards to methodology, we first plan to implement a feed-forward neural
network or support vector machine classifier as a baseline, transforming the
pixelated image into a single vector as input. We will first train our model
classifier to differentiate whether a person is wearing a mask or not. Then, we
will train our model to classify among the masked predictions whether a person
is wearing the mask correctly or incorrectly. After implementing our baseline,
we plan to design a convolutional neural network, which has shown to be
effective in image recognition tasks, and evaluate using the same metrics.

For our dataset, we’ll be using a set of 853 images from
[Kaggle](https://www.kaggle.com/andrewmvd/face-mask-detection) which each have
labels corresponding to wearing a mask, not wearing a mask, and wearing a mask
incorrectly, as well as bounding boxes around the masks. Some images have
multiple individuals, which offers valuable data points. If we find that this
dataset is too small for effective training, we will try to implement either
transfer learning or data augmentation techniques to work around this;
otherwise, we could simply collect more data. Given the recency of COVID-19
pandemic, there hasn’t been too much existing research on face mask detection,
but one recent
[study](https://www.sciencedirect.com/science/article/pii/S0263224120308289) has
shown that machine learning models are definitely capable of achieving high
performance with regards to this task. This study will likely guide some of our
design choices throughout this project.

For our experiments, we plan on partitioning our dataset into three sets: a
training set, a validation set, and test set. It is likely that our dataset will
need significant data augmentation (mirroring, brightness adjustments, etc.) in
order to suffice for training. Our model’s performance will be evaluated based
on the percentage of images it is able to classify correctly into correctly
wearing a mask, incorrectly wearing a mask, or not wearing a mask at all. We can
use simple precision and recall metrics to measure relative improvements over
time or between our baseline and main models.

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Here's a blank template to get started:
**To avoid retyping too much info. Do a search and replace with your text editor for the following:**
`github_username`, `repo_name`, `twitter_handle`, `email`


### Built With

* []()
* []()
* []()

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
```sh
npm install npm@latest -g
```

### Installation

1. Clone the repo
```sh
git clone https://github.com/dastratakos/Face-Mask-Detection.git
```
2. Install NPM packages
```sh
npm install
```

<!-- USAGE EXAMPLES -->
## Usage

1. Download dataset from Kaggle.
2. Set up `virtualenv`.
3. Crop the images.
```
python crop.py
```
4. Run data augmentation
```
python augment.py
```
5. Run the pipeline.
```
python run_pipeline.py
```

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/dastratakos/Face-Mask-Detection/issues) for a list of proposed features (and known issues).

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the Stanford License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email

Project Link: [https://github.com/dastratakos/Face-Mask-Detection](https://github.com/dastratakos/Face-Mask-Detection)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* []()
* []()
* []()

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/dastratakos/Face-Mask-Detection.svg?style=flat-square
[contributors-url]: https://github.com/dastratakos/Face-Mask-Detection/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/dastratakos/Face-Mask-Detection.svg?style=flat-square
[forks-url]: https://github.com/dastratakos/Face-Mask-Detection/network/members
[stars-shield]: https://img.shields.io/github/stars/dastratakos/Face-Mask-Detection.svg?style=flat-square
[stars-url]: https://github.com/dastratakos/Face-Mask-Detection/stargazers
[issues-shield]: https://img.shields.io/github/issues/dastratakos/Face-Mask-Detection.svg?style=flat-square
[issues-url]: https://github.com/dastratakos/Face-Mask-Detection/issues
[license-shield]: https://img.shields.io/github/license/dastratakos/Face-Mask-Detection.svg?style=flat-square
[license-url]: https://github.com/dastratakos/Face-Mask-Detection/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/dean-stratakos-8b338b149
[product-screenshot]: images/screenshot.png
