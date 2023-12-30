# Code for the paper _Investigating Gender Bias in News Summarization_

This repository currently only contains the code for template generation. Evaluation code will follow soon.

## Generating input

Download the proper names from the [BBQ repository](https://github.com/nyu-mll/BBQ/blob/main/templates/vocabulary_proper_names.csv) and put them in ``data/``.

To generate input examples run ``python -m summarybias.instancegen.generate --onto-path [path to the OntoNotes nw directory]``.
You will see some warnings about overlapping mentions. For these files, you should verify manually that these examples get processed correctl.