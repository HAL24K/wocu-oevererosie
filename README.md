# wocu-oevererosie
River bank erosion project

## To run notebooks

```bash
pipenv install
pipenv run jupyter lab
```

## Code philosophy

The aim of this project is to end up with  product-like codebase that would be almost ready to deploy to production.
As such, the potential user should not be expected to run anything manually and only communicate with the codebase via
configuration files and a handful of entry points.

At the highest general level, the user provides the data and - via some configuration - provides details what is
to be done with the data. The code then processes these data, including transforming them, adding external data, and
creating features for the model input. The - separately trained - model then predicts the variables, the output
of which gets processed into the required output and returned to the user.

The data processing and feature creation is all done by the `DataHandler` class, which lives 
in `src/data/data_handler.py`. Its input is a configuration object and, possibly, a set of GeoDataFrames (the main 
class in the `geopandas` library). Various methods in `DataHandler` then process and enrich the data, and create
the features. The results of these operations are stored as various attributes of the existing `DataHandler` object.
The `DataHandler` object uses a set of known operations to turn geospatial data - local or external - into features.
These are specified in the configuration object and can be seen in `src/data/feature_generation_configuration.py`. Each
of the operations should map to a feature-generating method, e.g. in `src/utils.py`. See the `_generate_region_features`
method in `DataHandler` for an example of how to use these operations.

The `DataCollector` class, in `src/data/data_collector.py`, is responsible for collecting the external data from WFS.
The known services are provided in a configuration (see `src/config.py`). The `DataCollector` object is provided with
a source shape - a small part of an area - inside of which it collects and returns various data contained in WFS (more
precisely, it collects the data from the bounding box of the source shape). These data are then returned 
as geodataframes and features are calculated from them.


## Repository structure

`data/` - contains the data used in the project. Ideally nothing is commited here. If some data ships with the project,
it would appear in a different place.

`notebooks/` - contains notebooks created in the project. These are ideally all self-contained and don't contain any 
code that adds user-relevant functionality to the product itself. The notebooks also don't contain any output, 
so that we don't bloat the repository - always make sure to clear the cell output and save your notebook before 
committing.

`scripts/` - contains helper scripts in the project. Similar to `notebooks`, no user-relevant functionality should be
contained in here. No tests are required.

`src/` - contains the source code for the project itself. Make sure to follow good coding practices when adding
functionality and always include unit tests. 

`tests/` - contains the unit and other tests for the project. All `test_*.py` files here will be run by the pipeline; 
we use the `pytest` framework for testing. Data used in testing live in `tests/assets/` - if adding anything, make sure
it's not too big (it's pushing to git)


## Good coding practices

Follow general good coding practices, including, but very much not limited to:

- DRY ("Don't Repeat Yourself") - if you are using the same functionality in multiple places, refactor it to a function 
- don't hardcode anything you can avoid. Define constants and configurations in a single place, for instance 
  `src/config.py`, `src/constants.py` etc.
- variable and function names should be descriptive. Long variable/funciton names are ok, better than short
  or jargony ones. `default_river_depth_near_sea` is better than `drdns`.
- use automated checks. At least run `black scripts/ src/ tests/` before each commit - if some changes are leftover,
  the pipeline will fail.
- don't commit any passwords or other secrets
- always work on a feature branch and before merging make sure that all tests pass and someone else reviewed
  your changes
- make errors, warnings and other messages verbose, informative and ideally leading the user to the solution. Don't say
  `Unknown input.`, specify `Unknown fruit: 'dorian'. Please pick one of: 'apple', 'banana', 'cherry'.` Guide the user,
  don't just say `Bad input dimension: 4`, but rather `Bad input dimension: 3. This typically happens when you accidentally label "year" as category in the configuration.` or something.
- Log more than you'll need. Keep being verbose and don't shy away from interpreting the logs for the user:
  `Number of data samples: 1000; number of samples in features: 900. We expect some samples to have missing data and thus be dropped from the feature calcualtion.`
  Be reassuring where you can.

## TODOs

- features from the local data
- training/evaluation/"run_model" scripts
- address TODOs in the code
- add commit hooks
