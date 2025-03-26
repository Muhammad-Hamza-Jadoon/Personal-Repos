# Verse-AI

<div align="center">
	<img src="images/logo.png" height="300">
</div>

<br>

<div align="center">
   <img src="images/made-with-python.svg">
   <!-- <img src="images/built-with-love.svg"> -->
   <br>
   <img src="images/WIP.svg"><img src="images/contribute.svg">
</div>

## 📖 Overview

Lyrical is an end-to-end Lyrics Generation Model that leverages the power of recurrent neural networks (RNNs) and natural language processing to learn the patterns, structure, and vocabulary of song lyrics. By training on a diverse collection of lyrics from different genres and artists, the model can capture the nuances and characteristics of various music styles and intelligently generate lyrics that adhere to specific themes, genres, or even mimic the style of a particular artist.

## ➕ Features

- **Data Collection**: The model incorporates a web scraping component to collect song lyrics from popular online platforms such as Genius Lyrics. The lyrics, along with associated metadata such as artist name, song title, genre, and album, are gathered to provide additional context for conditioning the model during the generation process.
- **Data Preprocessing**: The collected lyrics are meticulously preprocessed to ensure data quality and consistency. This includes removing irrelevant information, handling missing values, standardizing text formatting, and performing other data cleaning tasks. The lyrics are also tokenized and converted into numerical representations suitable for training the lyrics generation model.
- **Conditioning and Context**: The model takes advantage of the additional metadata, such as genre, artist name, and album, to condition the generated lyrics. By providing these conditioning factors during the training and generation phases, the model can generate lyrics that align with specific styles, themes, or artists, producing more contextually relevant and coherent results.
- **Model Training**: The lyrics generation model employs advanced deep learning techniques, such as recurrent neural networks (RNNs) or transformer architectures, to learn the underlying patterns and structures within the lyrics dataset. The model is trained using techniques like sequence-to-sequence modeling or language modeling approaches, optimizing it to generate meaningful and creative lyrics.
- **Evaluation and Fine-tuning**: The generated lyrics can be evaluated using metrics such as perplexity, BLEU score, or through human evaluation. This evaluation process helps fine-tune the model by iteratively adjusting hyperparameters, experimenting with different architectures, and training strategies to optimize the results and enhance the quality and creativity of the generated lyrics.
- **Lyrics Generation**: Once trained, the model can generate new lyrics by providing a seed input or prompt along with the desired conditioning information. The model uses the learned patterns and structures from the training dataset to generate lyrics that are coherent, stylistically appropriate, and aligned with the provided conditioning factors.
