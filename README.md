# Code for the paper _Gender Bias in News Summarization: Measures, Pitfalls and Corpora_

## Obtaining Ontonotes

Due to the license of [OntoNotes](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjcs5KYj6uEAxXIh_0HHSvQDMQQFnoECBQQAQ&url=https%3A%2F%2Fcatalog.ldc.upenn.edu%2FLDC2013T19&usg=AOvVaw2ACE0tDhYedJ_nP8we1HPi&opi=89978449), we can provide neither complete templates, nor generated summaries which might contain verbatim copies of input documents. However, we provide code to reproduce our experiments if you have access to OntoNotes data.

## Generating input

Download the proper names from the [BBQ repository](https://github.com/nyu-mll/BBQ/blob/main/templates/vocabulary_proper_names.csv) and put them in ``data/``.

To generate input examples run ``python -m summarybias.instancegen.generate --onto-path [path to the OntoNotes nw directory]``.
You will see some warnings about overlapping mentions. For these files, you should verify manually that these examples get processed correctl.


## Generating Summaries

Instances for evaluation should be in the same format as the inputs and contain all of their keys, along with a new key ''summary'' (see the code in summarybias.summarize).

## Evaluation

The score computation requires summaries to be parsed. summarybias.evaluate.parse will parse the summary and save it in ``parsed_summaries/``. You can then evaluate using ``scripts/evaluate_gender_file.py parsed_summaries/[filename]`` (don't forget to set the ``PYTHONPATH``).