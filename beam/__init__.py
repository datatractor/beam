"""
This script is intended as an example usage and reference implementation of the
API endpoints exposed on the |yardsite|_. Currently, it can be used to:

- query the registry of `Extractors <https://yard.datatractor.org/extractors/>`_ for
  extractors that support a given file type,
- install those extractors in a fresh Python virtual environment environment via `pip`,
- invoke the extractor either in Python or at the CLI, producing Python objects or files
  on disk.

.. |yardsite| image:: https://badgen.net/static/%F0%9F%9A%9C%20datatractor/yard

.. _yardsite: https://yard.datatractor.org/

"""

import argparse
import importlib.metadata
import json
import multiprocessing.managers
import multiprocessing.shared_memory
import pickle
import platform
import re
import subprocess
import urllib.error
import urllib.request
import venv
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional

__all__ = ("extract", "Extractor")
__version__ = importlib.metadata.version("datatractor-beam")

REGISTRY_BASE_URL = "https://yard.datatractor.org/api"
BIN = "Scripts" if platform.system() == "Windows" else "bin"


class SupportedExecutionMethod(Enum):
    # TODO: would be nice to generate these directly from the LinkML schema
    CLI = "cli"
    PYTHON = "python"


class SupportedInstallationMethod(Enum):
    PIP = "pip"
    CONDA = "conda"


class SupportedUsageScope(Enum):
    META = "meta-only"
    DATA = "meta+data"


def run_beam():
    argparser = argparse.ArgumentParser(
        prog="beam",
        description="""CLI for datatractor extractors that takes a filename and a filetype, then installs and runs an appropriate extractor, if available, from the chosen registry (default: https://registry.datatractor.org/). Filetype IDs can be found in the registry API at e.g., https://registry.datatractor.org/api/filetypes. If a matching extractor is found at https://registry.datatractor.org/api/extractors, it will be installed into a virtual environment local to the beam installation. The results of the extractor will be written out to a file at --outfile, or in the default location for that output file type.""",
    )

    argparser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s version {__version__}",
    )

    argparser.add_argument(
        "filetype",
        help="FileType.ID of the input file",
        default=None,
    )

    argparser.add_argument(
        "infile",
        help="Path of the input file",
        default=None,
    )

    argparser.add_argument(
        "--outfile",
        "-o",
        help="Optional path of the output file",
        default=None,
    )

    args = argparser.parse_args()

    extract(
        input_path=args.infile,
        input_type=args.filetype,
        output_path=args.outfile,
        preferred_mode=SupportedExecutionMethod.CLI,
    )


def extract(
    input_path: Path | str,
    input_type: str,
    output_path: Path | str | None = None,
    output_type: str | None = None,
    preferred_mode: SupportedExecutionMethod | str = SupportedExecutionMethod.PYTHON,
    preferred_scope: SupportedUsageScope | str = SupportedUsageScope.DATA,
    install: bool = True,
    use_venv: bool = True,
    extractor_definition: dict | None = None,
    registry_base_url: str = REGISTRY_BASE_URL,
) -> Any:
    """Parse a file given its path and file type.

    Parameters:
        input_path: The path or URL of the file to parse.
        input_type: The ID of the ``FileType`` in the registry.
        output_path: The path to write the output to.
            If not provided, the output will be requested to be written
            to a file with the same name as the input file, but with an extension as
            defined using the ``output_type``. Defaults to ``{input_path}.out``.
        output_type: A string specifying the desired output type.
        preferred_mode: The preferred execution method.
            If the extractor supports both Python and CLI, this will be used to determine
            which to use. If the extractor only supports one method, this will be ignored.
            Accepts the ``SupportedExecutionMethod`` values of "cli" or "python".
        install: Whether to install the extractor package before running it. Defaults to True.
        extractor_definition: A dictionary containing the extractor definition to use instead
            of a registry lookup.
        registry_base_url: The base URL of the registry to use. Defaults to the
            |yardsite|_.

    Returns:
        The output of the extractor, either a Python object or nothing.

    """
    tmp_path: Optional[Path] = None
    try:
        if isinstance(input_path, str) and re.match("^http[s]://*", input_path):
            _tmp_path, _ = urllib.request.urlretrieve(input_path)
            tmp_path = Path(_tmp_path)
            input_path = tmp_path

        input_path = Path(input_path)

        if not input_path.exists():
            raise RuntimeError(f"File {input_path} does not exist")

        output_path = Path(output_path) if output_path else None

        if isinstance(preferred_mode, str):
            preferred_mode = SupportedExecutionMethod(preferred_mode)
        if isinstance(preferred_scope, str):
            preferred_scope = SupportedUsageScope(preferred_scope)

        if extractor_definition is None:
            try:
                request_url = f"{registry_base_url}/filetypes/{input_type}"
                response = urllib.request.urlopen(request_url)
            except urllib.error.HTTPError as e:
                raise RuntimeError(
                    f"Could not find file type {input_type!r} in the registry at {request_url!r}.\nFull error: {e}"
                )
            json_response = json.loads(response.read().decode("utf-8"))
            extractors = json_response["data"]["registered_extractors"]
            if not extractors:
                raise RuntimeError(
                    f"No extractors found for file type {input_type!r} in the registry"
                )
            elif len(extractors) > 0:
                print(f"Discovered the following extractors: {extractors}.")
        else:
            extractors = [extractor_definition]

        matching_definition = None
        for extractor in extractors:
            if isinstance(extractor, str):
                try:
                    request_url = f"{registry_base_url}/extractors/{extractor}"
                    entry = urllib.request.urlopen(request_url)
                except urllib.error.HTTPError as e:
                    raise RuntimeError(
                        f"Could not find extractor {extractor!r} in the registry at {request_url!r}.\nFull error: {e}"
                    )
                extractor = json.loads(entry.read().decode("utf-8"))["data"]

            match = find_matching_usage(
                usages=extractor["usage"],
                input_type=input_type,
                preferred_mode=preferred_mode,
                preferred_scope=preferred_scope,
                strict=True,
            )
            if match is not None:
                print(f"Found matching usage with extractor: {extractor['id']!r}")
                matching_definition = extractor
                break

        if matching_definition is None:
            raise RuntimeError(
                "No extractors found with the preferred execution mode and input type."
            )
        plan = ExtractorPlan(
            matching_definition,
            preferred_mode=preferred_mode,
            install=install,
            use_venv=use_venv,
        )

        return plan.execute(
            input_type=input_type,
            input_path=input_path,
            output_type=output_type,
            output_path=output_path,
        )
    finally:
        if tmp_path:
            tmp_path.unlink()


def find_matching_usage(
    usages: list[dict],
    input_type: str,
    preferred_mode: SupportedExecutionMethod = SupportedExecutionMethod.PYTHON,
    preferred_scope: SupportedUsageScope = SupportedUsageScope.DATA,
    strict: bool = False,
) -> dict:
    """
    Find the usage among usages that best matches the requirements. If strict is True,
    all criteria must be fulfilled.

    Returns the usage that best matches the requirements, following these criteria:

     - The matching usage must match the ``input_type``; if there is no usage that
       matches, ``None`` is returned.
     - The matching usage should match the preferred execution method (``preferred_mode``).
       If there is no such usage, a warning is raised, but a usage is still returned.
     - The matching usage should match the preferred extraction scope (``preferred_scope``).
       If there is no such usage, a warning is raised, but a usage is still returned.

    Raises:
        RuntimeError if no matching usage is found.

    """
    candidates: dict[str, Any] = {"mode": None, "scope": None}
    for usage in usages:
        if (
            usage["supported_filetypes"] is None
            or input_type in usage["supported_filetypes"]
        ):
            thisMethod = SupportedExecutionMethod(usage["method"])
            if (
                usage["scope"] is None
            ):  # HACK - default should be "meta+data" -- can remove after schema 1.0.2
                usage["scope"] = "meta+data"
            thisScope = SupportedUsageScope(usage["scope"])
            if thisMethod == preferred_mode and thisScope == preferred_scope:
                return usage
            elif thisMethod == preferred_mode:
                candidates["mode"] = usage
            elif thisScope == preferred_scope:
                candidates["scope"] = usage
    if not strict and candidates["mode"] is not None:
        print("Found usage with matching execution method but wrong extraction scope.")
        return candidates["mode"]
    if not strict and candidates["scope"] is not None:
        print("Found usage with matching extraction scope but wrong execution method.")
        return candidates["scope"]

    raise RuntimeError(
        f"Found no matching usage for input_type {input_type!r} with {preferred_mode} and {preferred_scope}"
    )


class ExtractorPlan:
    """A plan for parsing a file."""

    entry: dict
    """The registry entry to use for parsing."""

    def __init__(
        self,
        entry: dict,
        install: bool = True,
        preferred_mode: SupportedExecutionMethod = SupportedExecutionMethod.PYTHON,
        preferred_scope: SupportedUsageScope = SupportedUsageScope.DATA,
        use_venv: bool = True,
    ):
        """Initialize the plan, optionally installing the specific parser package."""
        self.entry = entry
        self.preferred_mode = preferred_mode
        self.preferred_scope = preferred_scope

        if use_venv:
            self.venv_dir: Path | None = (
                Path(__file__).parent.parent / "beam-venvs" / f"env-{self.entry['id']}"
            )
            venv.create(self.venv_dir, with_pip=True)
        else:
            self.venv_dir = None

        if install:
            self.install()

    def install(self):
        """Follows the installation instructions for the entry.

        Currently supports the following :doc:`datatractor_schema:datatractor_schema/InstallerTypes`

           - ``"pip"``

        The installation proceeds inside the appropriate venv, if configured.
        """
        print(f"Attempting to install {self.entry.get('id', self.entry)}")
        if not self.entry.get("installation"):
            raise RuntimeError(
                "No installation instructions provided for {self.entry.get('id', self.entry)}"
            )

        for instructions in self.entry["installation"]:
            method = SupportedInstallationMethod(instructions["method"])
            if method == SupportedInstallationMethod.PIP:
                try:
                    for p in instructions["packages"]:
                        command = [
                            (
                                str(self.venv_dir / BIN / "python")
                                if self.venv_dir
                                else "python"
                            ),
                            "-m",
                            "pip",
                            "install",
                            f"{p}",
                        ]
                        subprocess.run(command, check=True)
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(
                    f"Installation method {instructions['method']} not yet supported"
                )

    def execute(
        self,
        input_type: str,
        input_path: Path,
        output_type: str | None = None,
        output_path: Path | None = None,
    ):
        """Follows the required usage instructions for the entry.

        Currently supports the following :doc:`datatractor_schema:datatractor_schema/UsageTypes`:

          - `"cli"`
          - `"python"`

        The execution proceeds in the appropriate venv, if configured.
        """
        template = None
        for filetype in self.entry["supported_filetypes"]:
            if filetype["id"] == input_type:
                template = filetype.get("template")
                break
        else:
            raise ValueError(
                f"File type {input_type!r} not supported by {self.entry['id']!r}"
            )

        method, command, setup = self.parse_usage(
            self.entry["usage"],
            input_type,
            preferred_mode=self.preferred_mode,
            preferred_scope=self.preferred_scope,
        )

        if output_path is None:
            suffix = ".out" if output_type is None else f".{output_type}"
            output_path = input_path.with_suffix(suffix)

        command = self.apply_template_args(
            command,
            method,
            input_type=input_type,
            input_path=input_path,
            output_type=output_type,
            output_path=output_path,
            additional_template=template,
        )

        if setup is not None:
            setup = self.apply_template_args(
                setup,
                method,
                input_type=input_type,
                input_path=input_path,
                output_type=output_type,
                output_path=output_path,
                additional_template=template,
            )

        if method == SupportedExecutionMethod.CLI:
            print(f"Executing {command}")
            if self.venv_dir:
                venv_bin_command = str(self.venv_dir / BIN / command)
                output = self._execute_cli_venv(venv_bin_command)
            else:
                output = self._execute_cli(command)

            if not output_path.exists():
                raise RuntimeError(
                    f"Requested output file {output_path} does not exist"
                )

            print(f"Wrote output to {output_path}")

        elif method == SupportedExecutionMethod.PYTHON:
            if self.venv_dir:
                output = self._execute_python_venv(command, setup)
            else:
                output = self._execute_python(command, setup)

        return output

    def _execute_cli(self, command):
        print(f"Executing {command=}")
        results = subprocess.check_output(command, shell=True)
        return results

    def _execute_cli_venv(self, command):
        print(f"Executing {command=} in venv")
        py_cmd = f"import subprocess; subprocess.check_output(r'{command}', shell=True)"
        command = [str(self.venv_dir / BIN / "python"), "-c", py_cmd]
        results = subprocess.check_output(command)
        return results

    @staticmethod
    def _prepare_python(command) -> tuple[list[str], list[str], dict]:
        function_tree = command.split("(")[0].split(".")
        # Treat contents of first brackets as arguments
        # TODO: this gets a bit gross with more complex arguments with nested brackets.
        # This parser will need to be made very robust

        segments = command.split("(")[1].split(")")[0].split(",")
        kwargs = {}
        args = []

        def dequote(s: str):
            s = s.strip()
            if s.startswith("'") or s.endswith("'"):
                s = s.removeprefix("'")
                s = s.removesuffix("'")
            elif s.startswith('"') or s.endswith('"'):
                s = s.removeprefix('"')
                s = s.removesuffix('"')
            return s.strip()

        def _parse_python_arg(arg: str):
            if "=" in arg:
                split_arg = arg.split("=")
                if len(split_arg) > 2 or "{" in arg or "}" in arg:
                    raise RuntimeError(f"Cannot parse {arg}")

                return {dequote(arg.split("=")[0]): dequote(arg.split("=")[1])}
            else:
                return dequote(arg)

        for arg in segments:
            parsed_arg = _parse_python_arg(arg)
            if isinstance(parsed_arg, dict):
                kwargs.update(parsed_arg)
            else:
                args.append(parsed_arg)

        return function_tree, args, kwargs

    def _execute_python_venv(self, entry_command: str, setup: str):
        data = None

        with multiprocessing.managers.SharedMemoryManager() as shmm:
            shm = shmm.SharedMemory(size=1024 * 1024 * 1024)
            py_cmd = (
                f'print("Launching extractor"); import {setup}; import pickle; import multiprocessing.shared_memory;'
                + f"shm = multiprocessing.shared_memory.SharedMemory(name={shm.name!r});"
                + f"data = pickle.dumps({entry_command}); shm.buf[:len(data)] = data; print('Done!');"
            )

            if not self.venv_dir:
                raise RuntimeError("Something has gone wrong; no `venv_dir` set")

            command = [str(self.venv_dir / BIN / "python"), "-c", py_cmd]
            subprocess.check_output(
                command,
            )
            data = pickle.loads(shm.buf)

        return data

    def _execute_python(self, command: str, setup: str):
        from importlib import import_module

        if " " not in setup:
            module = setup
        else:
            raise RuntimeError("Only simple `import <setup>` invocation is supported")

        extractor_module = import_module(module)

        function_tree, args, kwargs = self._prepare_python(command)

        def _descend_function_tree(module: ModuleType, tree: list[str]) -> Callable:
            if tree[0] != module.__name__:
                raise RuntimeError(
                    "Module name mismatch: {module.__name__} != {tree[0]}"
                )
            _tree = tree.copy()
            _tree.pop(0)
            function: Callable | ModuleType = module
            while _tree:
                function = getattr(function, _tree.pop(0))
            return function  # type: ignore

        try:
            function = _descend_function_tree(extractor_module, function_tree)
        except AttributeError:
            raise RuntimeError(f"Could not resolve {function_tree} in {module}")

        return function(*args, **kwargs)

    @staticmethod
    def apply_template_args(
        command: str,
        method: SupportedExecutionMethod,
        input_type: str,
        input_path: Path,
        output_type: str | None = None,
        output_path: Path | None = None,
        additional_template: dict | None = None,
    ) -> str:
        """Reference implementation of templating in the Datatractor Schema.

        See the :doc:`datatractor_schema:datatractor_schema/UsageTemplate` for details
        of the individual arguments.

        Parameters:
            command: The command to apply the template to.
            method: The execution method to use.
            input_type: The input type to use.
            input_path: The input path to use.
            output_type: The output type to use.
            output_path: The output path to use.
            additional_template: Additional template arguments to use, which
                overwrite the default arguments.

        Returns:
            The templated command.

        """
        if additional_template is None:
            additional_template = {}
        default_fields = {"input_type", "input_path", "output_type", "output_path"}
        for field in default_fields:
            value = additional_template.get(field) or locals()[field]
            if value is None:
                continue
            if method == SupportedExecutionMethod.CLI:
                command = command.replace(f"{{{{ {field} }}}}", str(value))
            else:
                command = command.replace(f"{{{{ {field} }}}}", f"{str(value)!r}")

        return command

    @staticmethod
    def parse_usage(
        usages: list[dict],
        input_type: str,
        preferred_mode: SupportedExecutionMethod = SupportedExecutionMethod.PYTHON,
        preferred_scope: SupportedUsageScope = SupportedUsageScope.DATA,
    ) -> tuple[SupportedExecutionMethod, str, str]:
        usage = find_matching_usage(usages, input_type, preferred_mode, preferred_scope)
        if not usage:
            raise RuntimeError(
                f"No matching usage found for {input_type} with {preferred_mode} and {preferred_scope}"
            )

        method = usage["method"]
        command = usage["command"]
        setup = usage.get("setup", None)
        return SupportedExecutionMethod(method), command, setup
