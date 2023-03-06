import sys

import click

import rombus.core as core
import rombus.plot as plot

from rombus.model import init_model

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
FLEX_CONTEXT_SETTINGS = dict(
    help_option_names=["-h", "--help"],
    ignore_unknown_options=True,
    allow_extra_args=True,  # needed for the passing of model parameters
)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.argument("model", type=str)
@click.pass_context
def cli(ctx, model):
    """Perform greedy algorythm operations with Rombus"""

    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)

    # Import user code defining the model we are working with
    ctx.obj = init_model(model)


@cli.command(context_settings=FLEX_CONTEXT_SETTINGS)
@click.argument("parameters", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def evaluate(ctx, parameters):
    """Evaluate a reduced order model and compare it to truth

    PARAMETERS is a list of parameter values of the form A=VAL B=VAL ..."""

    # The model gets passed as context
    model = ctx.obj

    # Parse the model parameters, which should have been given as arguments
    model_params = model.parse_cli_params(parameters)

    # Generate plot
    plot.compare_rom_to_true(model, model_params)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("filename_in", type=click.Path(exists=True))
@click.option(
    "--do_step",
    type=click.Choice(["RB", "EI"], case_sensitive=False),
    default=None,
    help="Do only one step: RB=reduced basis; EI=empirical interpolant",
)
@click.pass_context
def build(ctx, filename_in, do_step):
    """Build a reduced order model

    FILENAME_IN is the 'greedy points' numpy or csv file to take as input
    """

    # The model gets passed as context
    model = ctx.obj

    # Compute the reduced basis
    reduced_basis = None
    if not do_step or do_step == "RB":
        samples = core.Samples(model, filename=filename_in)
        reduced_basis = core.ReducedBasis(model, samples)

    # Compute the empirical interpolant
    if not do_step or do_step == "EI":
        # Still need to implement reading of reduced_basis for this
        if reduced_basis is None:
            raise Exception
        core.EmpiricalInterpolant(reduced_basis)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("filename_in", type=click.Path(exists=True))
@click.pass_context
def refine(ctx, filename_in):
    """Refine parameter sampling to impove a reduced order model"""

    model = ctx.obj

    # Load samples from file
    samples = core.Samples(model, filename=filename_in)

    # Build model and refine it
    ROM = core.ROM(model, samples).refine()

    # Write results
    print(ROM)


if __name__ == "__main__":
    cli(obj={})
    sys.exit()
