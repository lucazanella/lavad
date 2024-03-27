<div align="center">

[![Website](https://img.shields.io/badge/website-LAVAD-4285F4.svg)](https://lucazanella.github.io/lavad/)

# Harnessing Large Language Models for Training-free Video Anomaly Detection

Luca Zanella, [Willi Menapace](https://www.willimenapace.com/), [Massimiliano Mancini](https://mancinimassimiliano.github.io/), [Yiming Wang](https://www.yimingwang.it/), [Elisa Ricci](https://eliricci.eu/) <br>

</div>

<p align="center">
  <img style="width: 75%" src="media/teaser.png">
</p>
<br>

> **Abstract:** *Video anomaly detection (VAD) aims to temporally locate abnormal events in a video. Existing works mostly rely on training deep models to learn the distribution of normality with either video-level supervision, one-class supervision, or in an unsupervised setting.  Training-based methods are prone to be domain-specific, thus being costly for practical deployment as any domain change will involve data collection and model training. In this paper, we radically depart from previous efforts and propose **LA**nguage-based **VAD** (LAVAD), a method tackling VAD in a novel, *training-free* paradigm, exploiting the capabilities of pre-trained large language models (LLMs) and existing vision-language models (VLMs). We leverage VLM-based captioning models to generate textual descriptions for each frame of any test video. With the textual scene description, we then devise a prompting mechanism to unlock the capability of LLMs in terms of temporal aggregation and anomaly score estimation, turning LLMs into an effective video anomaly detector. We further leverage modality-aligned VLMs and propose effective techniques based on cross-modal similarity for cleaning noisy captions and refining the LLM-based anomaly scores. We evaluate LAVAD on two large datasets featuring real-world surveillance scenarios (UCF-Crime and XD-Violence), showing that it outperforms both unsupervised and one-class methods without requiring any training or data collection.*

## Code and data will be released soon.
