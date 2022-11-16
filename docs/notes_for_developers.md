# Notes for Developers

## Versioning

Semantic versioning (i.e. a scheme that follows a `vMAJOR.MINOR.PATCH` format; see <https://semver.org> for details) is used for this project.  The single point of truth for the current production version is the last git tag on the main branch with a `v[0-9]*` format.

Changes to `PATCH` are handled by a GitHub workflow which increments this value and creates a new tag whenever a push occurs to the `main` branch.  This ensures that every commit on the `main` branch is assigned a unique version.  If a change to `MINOR` (for backward-compatible functionality changes) or to `MAJOR` (for breaking changes) is required then a new version tag should be manually added to the `main` branch through the GitHub UI.  Presumably, any such change will warrent a new release, and this is most easily achieved by creating a new tag (rather than selecting the last - automatically generated - version tag) when the release is created.

## Releases

Releases are generated through the GitHub UI.  A GitHub workflow has been configured to do the following when a new release is produced:

* Run the tests for the project
* Ensure that the project builds
* Publish a new version of the code on [PyPI](https://pypi.org/).
* Rebuild the documentation on *Read the Docs*

To generate a new release, do the following:

* Navigate to the project's GitHub page, 
* Click on `Releases` in the sidebar,
* Click on `Create a new release` if this is the first release you have generatede, or `draft release` if this is a subsequent release,
* If this release is a PATCH (i.e. no new features or breaking changes) then click on `Choose a tag` and select the most recent version listed; otherwise, if it is a new feature with no breaking
	changes, create a new tag with the format `vMAJOR.MINOR.0`, where `MINOR` is incremented by 1 from the last version tag; otherwise - in the case of a breaking change - create a new tag with the
	format `vMAJOR.0.0`, where `MAJOR` is incremented by 1 from the last version tag,
* Write some text describing the release, and
* Click `Publish Release`.

If your accounts and the repository are all properly configured and all goes well (tests are passed, etc.), then the following will happen:

* a new *GitHub* release will be generated;
* the release will be published on *PyPI*; and
* the documentation will be rebuilt on *Read the Docs*.

## Development Environment Set-up

This section details how to grab a copy of this code and configure it for development purposes.  In what follows, we will assume that you have already created *GitHub* and *Read the Docs* accounts for this purpose.  If not, first visit  <https://github.com> and/or <https://readthedocs.org> respectively to do so.

### The Code

A local copy of the code can be configured as follows:

1. Create a fork: 
	* navigate to the GitHub page hosting the project and click on the `fork` button at the top of the page;
	* Edit the details you want to have for the new repoitory; and
	* Press `Create fork`,
	* Generate a local 

2. If you want to work from a local clone:
	* First grab a local copy of the code (e.g. `git clone <url>` where `<url>` can be obtained by clicking on the green `Code` button on the project GitHub page);
	* Create a new GitHub repository for the account to host the code by logging into *GitHub* and clicking the `+` selector at the very top and selecting `New repository`;
	* Edit the form detailing the new repository and click `Create repository` at the bottom;
	* Add the new *GitHub* repo as a remote repository to your local copy via `git remote add origin <newurl>`, where `<newurl>` can be obtained by clicking on the green `Code` button on the new repository's page; and
	* Push the code to the new GitHub repository with `git push origin main`.

### *GitHub*

Configure your *GitHub* repository following the directions [here](https://docs.readthedocs.io/en/stable/integrations.html#github).

### *Read the Docs*

Navigate to your RTD Project page and "Import a Project".  Your GitHub project with its new
Webhook should appear in this list.  Import it.

The documentation for this project should now build automatically for you when you generate a new release.

## Documentation

Documentation for this project is generated using [Sphinx](https://www.sphinx-doc.org/en/master/) and is hosted on *Read the Docs* for the latest release version.  Sphinx is configured here in a way which spares developers the pain of editing `.rst` files (the usual way of generating content for Sphinx).  Instead, `sphinx-apidoc` is used to generate `.rst` files from [Jinja2](https://jinja.palletsprojects.com/en/latest/) templates that have been placed in the `docs/_templates` directory of the project, which in turn source content from the project `README.md` file and [Markdown files](https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html) located in the `docs` directory of the project.

The structure of the resulting documentation will be as follows:

1. The project README file,
2. Markdown files from the `docs` directory, in the order specified in `root_doc.rst_t`, and
3. Content generated by `spinx-autodoc` from the docstrings of the project's Python code.

Documentation can be generated locally by running `make html` from the root directory of the code repository.  This will generate an html version of the documentation (in a new directory called `build`) which can be opened in your browser.  On a Mac, this can be done by running `open docs/index.html`, for example.  Note that you will need to have Sphinx installed for this (see the [Sphinx installation instructions](https://www.sphinx-doc.org/en/master/usage/installation.html) for details).

The majority of documentation changes can be managed in one of the following 4 ways:

1. **Edits to `README.md`**:

	Most high-level content should be presented in the `README.md` file.  This content gets used by the project documentation and is shared by the GitHub project page and the PyPI page.

2. **Project Docstrings**:

	Documentation for code changes specifying the codebase's API, implementation details, etc. should be managed directly in the Docstrings of the project's `.py` files.  This content will automatically be harvested by Sphinx.

3. **Existing Markdown files in the `docs` directory**:

	Examine the Markdown files in the `docs` directory.  Does the content that you want to add fit naturally within one of those files?  If so: add it there.

4. **Add a new Markdown file to the `docs` directory**:

	Otherwise, create a new `.md` file in the `docs` directory and add it to the list of Markdown files listed in the `docs/_templates/root_doc.rst_t` file.  Note that these files will be added to the documentation in the order specified, so place it in that list where you want it to appear in the final documentation.  Note that this new `.md` file should start with a top-level title (marked-up by starting a line with a single `#`; see the top of this file for an example).
