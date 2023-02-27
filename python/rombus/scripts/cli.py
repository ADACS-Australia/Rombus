import collections
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


def parse_cli_model_params(args, model):
    model_params = dict()
    for param_i in args:
        if not param_i.startswith("-"):
            res = param_i.split("=")
            if len(res) == 2:
                # NOTE: for now, all parameters are assumed to be floats
                model_params[res[0]] = float(res[1])
            else:
                raise click.ClickException(
                    f"Don't know what to do with argument '{param_i}'"
                )
        else:
            raise click.ClickException(f"Don't know what to do with option '{param_i}'")

    # Check that all parameters are specified and that they match what is
    # defined in the model
    assert collections.Counter(model_params.keys()) == collections.Counter(model.params)

    return model_params


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

    # Parse the model parameters, which should have been given as arguments
    model_params = parse_cli_model_params(parameters, ctx.obj)

    # Generate plot
    plot.compare_rom_to_true(ctx.obj, model_params)


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

    if not do_step or do_step == "RB":
        greedypoints, chunk_counts = core.read_divide_and_send_data_to_ranks(
            filename_in
        )
        core.make_reduced_basis(ctx.obj, greedypoints, chunk_counts)

    if not do_step or do_step == "EI":
        core.make_empirical_interpolant(ctx.obj)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("filename_in", type=click.Path(exists=True))
@click.pass_context
def refine(ctx, filename_in):
    """Refine parameter sampling to impove a reduced order model"""

    greedypoints, chunk_counts = core.read_divide_and_send_data_to_ranks(filename_in)

    core.refine(ctx.obj, greedypoints, chunk_counts)


if __name__ == "__main__":
    cli(obj={})
    sys.exit()
