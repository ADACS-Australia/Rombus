import sys

import click

import collections

import rombus.core as core
import rombus.plot as plot

from rombus.model import RombusModel

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
FLEX_CONTEXT_SETTINGS = dict(
    help_option_names=["-h", "--help"],
    ignore_unknown_options=True,
    allow_extra_args=True,  # needed for the passing of model parameters
)


class _OrderedGroup(click.Group):
    """This class is used for to ensure that the ordering of the CLI subcommands
    are in code-order in the CLI help and documentation."""

    def __init__(self, name=None, commands=None, **attrs):
        super(_OrderedGroup, self).__init__(name, commands, **attrs)
        #: the registered subcommands by their exported names.
        self.commands = commands or collections.OrderedDict()

    def list_commands(self, ctx):
        return self.commands


@click.group(cls=_OrderedGroup, context_settings=CONTEXT_SETTINGS)
@click.pass_context
def cli(ctx):
    """Perform greedy algorythm operations with Rombus"""

    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("model", type=str)
@click.argument("filename_samples", type=click.Path(exists=True))
@click.option(
    "--out",
    "-o",
    default="MODEL_BASENAME.hdf5",
    show_default=True,
    help="Output filename",
)
@click.option(
    "--do_step",
    type=click.Choice(["RB", "EI"], case_sensitive=False),
    default=None,
    help="Do only one step: RB=reduced basis; EI=empirical interpolant",
)
@click.pass_context
def build(ctx, model, filename_samples, out, do_step):
    """Build a reduced order model

    FILENAME_IN is the 'greedy points' numpy or csv file to take as input
    """

    # Load model
    model = RombusModel.load(model)

    # Load samples
    samples = core.Samples(model, filename=filename_samples)

    # Build ROM
    ROM = core.ROM(model, samples).build(do_step=do_step)

    # Write ROM
    if out == "MODEL_BASENAME.hdf5":
        filename_out = f"{model.model_basename}.hdf5"
    else:
        filename_out = out
    ROM.write(filename_out)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("filename_ROM", type=click.Path(exists=True))
@click.pass_context
def refine(ctx, filename_rom):
    """Refine parameter sampling to impove a reduced order model"""

    # Build model and refine it
    ROM = core.ROM.from_file(filename_rom).refine()

    # Write results
    filename_split = filename_rom.rsplit(".", 1)
    filename_out = f"{filename_split[0]}_refined.{filename_split[1]}"
    ROM.write(filename_out)


@cli.command(context_settings=FLEX_CONTEXT_SETTINGS)
@click.argument("filename_ROM", type=str)
@click.argument("parameters", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def evaluate(ctx, filename_rom, parameters):
    """Evaluate a reduced order model and compare it to truth

    PARAMETERS is a list of parameter values of the form A=VAL B=VAL ..."""

    # Read ROM
    ROM = core.ROM.from_file(filename_rom)

    # Parse the model parameters, which should have been given as arguments
    model_params = ROM.model.parse_cli_params(parameters)

    # Generate plot
    plot.compare_rom_to_true(ROM, model_params)


if __name__ == "__main__":
    cli(obj={})
    sys.exit()
