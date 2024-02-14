# Code for the paper _Gender Bias in News Summarization: Measures, Pitfalls and Corpora_

## Generating input

Download the proper names from the [BBQ repository](https://github.com/nyu-mll/BBQ/blob/main/templates/vocabulary_proper_names.csv) and put them in ``data/``.

To generate input examples run ``python -m summarybias.instancegen.generate --onto-path [path to the OntoNotes nw directory]``.
You will see some warnings about overlapping mentions. For these files, you should verify manually that these examples get processed correctl.


## Generating Summaries

Instances for evaluation should be in the same format as the inputs and contain all of their keys, along with a new key ''summary'' (see the code in summarybias.summarize).

## Evaluation

The score computation requires summaries to be parsed. summarybias.evaluate.parse will parse the summary and save it in ``parsed_summaries/``. You can then evaluate using ``scripts/evaluate_gender_file.py parsed_summaries/[filename]`` (don't forget to set the ``PYTHONPATH``).