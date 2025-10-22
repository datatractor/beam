<div align="center" style="padding-bottom: 1em;">
<img width="100px" align="center" src="https://avatars.githubusercontent.com/u/166528759">
</div>

# <div align="center">Datatractor Beam: <br/> Reference implementation of the Datatractor API</div>

<div align="center">


[![Documentation](https://badgen.net/badge/docs/datatractor.github.io/blue?icon=firefox)](https://datatractor.github.io/beam)
![Github status](https://badgen.net/github/checks/datatractor/beam/?icon=github)

</div>

Repository containing the reference implementation of the Datatractor API, published at [![Datatractor Yard](https://badgen.net/static/%F0%9F%9A%9Cdatatractor/yard)](https://yard.datatractor.org/).


For more information, see the publication:

> **Datatractor: Metadata, automation, and registries for extractor interoperability in the chemical and materials sciences**  
> Matthew L. Evans, Gian-Marco Rignanese, David Elbert & Peter Kraus  
> MRS Bulletin, 50, 838-845 (2025) [DOI: 10.1557/s43577-025-00925-8](https://link.springer.com/article/10.1557/s43577-025-00925-8)


or preprint at:

> **Datatractor: Metadata, automation, and registries for extractor interoperability in the chemical and materials sciences**  
> Matthew L. Evans, Gian-Marco Rignanese, David Elbert & Peter Kraus  
> [arXiv:2410.18839](https://arxiv.org/abs/2410.18839) (2024)

## `datatractor-beam` package

This repository contains a draft Python 3.10 package, located under the `./beam` directory.
The package can be used to:

- query the registry of [Extractors](https://yard.datatractor.org/) for extractors that support a given file type,
- install those extractors in a fresh Python virtual environment environment via `pip`,
- invoke the extractor either in Python or at the CLI, producing Python objects or files on disk.

### Installation

```shell
git clone git@github.com:datatractor/beam.git
cd beam
pip install .
```

### Usage

#### As a Python module
To extract data from a file, you can use the `extract` function from the `datatractor_beam` module inside your own Python code:

```python
from datatractor_beam import extract

# extract(<input_type>, <input_path>)
data = extract("./example.mpr",  "biologic-mpr")
```

This example will install the first extractor that is compatible with the `biologic-mpr` filetype that it finds in the registry. It will be installed into a fresh virtualenv (under `./beam-venvs`), and then executed on the file at `example.mpr`.

By default, the `extract` function will attempt to use the extractor's Python-based invocation (i.e. the optional `preferred_mode="python"` argument is specified). This means the extractor will be executed from within python, and the returned `data` object will be a Python object as defined (and supported) by the extractor. This may require additional packages to be installed, for examples `pandas` or `xarray`, which are both supported via the installation command `pip install .[formats]` above. If you encounter the following traceback, a missing "format" (such as `xarray` here) is the likely reason:

```
Traceback (most recent call last):
    [...]
    data = pickle.loads(shm.buf)
ModuleNotFoundError: No module named 'xarray'
```

Alternatively, if the `preferred_mode="cli"` argument is specified, the extractor will be executed using its command-line invocation. This means the output of the extractor will most likely be a file, which can be further specified using the `output_type` argument:

```python
from datatractor_beam import extract
ret = extract("example.mpr", "biologic-mpr", output_path="output.nc", preferred_mode = "cli")
```

In this case, the `ret` will be empty bytes, and the output of the extractor should appear in the `output.nc` file.

#### As a command line utility

The `datatractor` utility supports the following subcommands:

- `beam`: used to extract data from an input file of a known file type,
- `probe`: used to search the registry for extractors that match a known file type,
- `yard`: used to fetch the definition of an extractor from the registry, and
- `install`: used to install an extractor.

In particular, the `extract()` functionality discussed above can also be executed from the command line, implying `preferred_mode="cli"`. The command line invocation equivalent to the above Python syntax is:

```bash
datatractor beam biologic-mpr example.mpr --output-path output.nc
```


### Plans

- [x] Isolation of extractor environments
    - By installing each extractor into a fresh virtualenv, multiple extractors
      can be installed with possibly complex (and non-Python) dependencies.
    - This could be achieved by Python virtualenvs or Docker containers (or
      both!).
    - This will involve setting up a system for checking locally which
      extractors are available on a given machine.
    - Returning Python objects in memory will be tricker in this case, and would
      probably require choosing a few "blessed" formats that can be passed
      across subprocesses without any extractor specific classes,
      e.g., raw JSON/Python dicts, pandas dataframes or xarray datasets (as
      optional requirements, by demand).
- [x] A command-line for quickly running e.g., `beam <filename>`
- [ ] Extractor scaffold/template/plugin
    - If it can be kept similarly low-dependency, this package could also
      implement an extractor scaffold for those who want to modify existing
      extractors to follow the Datatractor API, and could automatically generate the
      appropriate registry entries for them.
- [ ] Testing and validation
    - We would like to move towards output validation, and this package would be
      the natural place to do so, again, perhaps supporting a few blessed
      formats, e.g., validating JSON output against an extractor-provided JSONSchema.
    - A testing mode that runs an extractor against all example files in the
      registry for that file type.
- [ ] File type detection following any rules added to the schemas
- [ ] Support for parallel processing
    - This package could handle invoking the same extractor on a large number of files.
- [ ] Support for other installation methods, such as `conda` and `docker`, to
  expand beyond purely `pip`-installable extractors.
