# Notes for Developers
This section provides details on how to configure and/or develop this codebase.

## Setting-up the Code
A local development copy of the code base can be configured with a remote repository as follows:

1. Forking/cloning from an established repository to create a local development copy:
	* navigate to the GitHub page hosting the project
    * If you want to fork the code so that you work on your own version of the repository:
        - click on the `fork` button at the top of the page;
	    - Edit the details you want to have for the new repoitory; and
	    - Press `Create fork`,
        - Navigate to the new repository
	* Obtain the URL for the repository you're going to use (denoted `<url>`) by clicking on the green `Code` button on the repository GitHub page;
	* Generate a local copy using `git clone <url>`;
    * Although not strictly necessary, it is recommended that you:
        - protect the main branch to only permit merges from pull requests.  This can be done by clicking on the 'branches' tab and clicking on the 'Protect this branch' button for the 'main' branch.
        - Select 'Require status checks to pass before merging' when you set-up this branch protection rule.

2. Creating a remote repository for an established local codebase:
	* Create a new GitHub repository for the account to host the code by
        - logging into *GitHub* and clicking the `+` selector at the very top and selecting `New repository`;
	    - Edit the form detailing the new repository and click `Create repository` at the bottom;
	* Obtain the URL for the repository you're going to use (denoted `<url>`) by clicking on the green `Code` button on the repository GitHub page;
	* Add the new *GitHub* repo as a remote repository to your local copy via `git remote add origin <url>`; and
	* Push the code to the new GitHub repository with `git push origin main`.

## Poetry

Poetry is used to manage this project.  Blah.

## Creating a Development Environment

Once a local copy is obtained, developers should make sure to create a Python environment for the project.  This can be done in one of several ways:
* Blah
* Blah

## Installing Development Dependencies

Development dependencies should be installed by moving to the project's root directory and executing the following:
``` console
$ poetry install --all-extras
```

In what follows, it will be assumed that this has been done.

## Configuring Services

This section details how to grab a copy of this code and configure it for development purposes.  Depending on your use case, you may require accounts with one or all of the following services:
* [__GitHub__](https://github.com),
* [__Read the Docs__](https://readthedocs.org), and
* The [__Python Package Index (PyPI)__](https://pypi.org).

In what follows, we will assume that you have already created these accounts, and need to configure a new remote repository (either because you have forked an established project or are initialising a new one).

### Configuring *GitHub*

Configure your *GitHub* repository following the directions [here](https://docs.readthedocs.io/en/stable/integrations.html#github).

You will also need to configure the following 'secrets' for the repository:
* **PYPI_TOKEN**,
* **RTD_WEBHOOK_TOKEN**, and
* **RTD_WEBHOOK_URL**

Once a webhook with *GitHub* has been established on *ReadtheDocs* (RTD), the value of **RTD_WEBHOOK_URL** can be found my migrating to the `Admin->Integrations` tab on the RTD project page.

This can be done by navigating to `Settings->Secrets->Actions`.  These need to be generated through the respective services.

### Configuring *Read the Docs*

Navigate to your RTD Project page and "*Import a Project*".  Your GitHub project with its new Webhook should appear in this list.  Import it.

The documentation for this project should now build automatically for you when you generate a new release.

### Configuring *PyPI*

Blah

## Guidelines

In the following we lay-out some guidelines for developing on this codebase.

### Branches

Development should not be conducted on the `main` branch.  Merges to this branch have been limited to Pull Requests (PRs) only.  Once a PR is opened for the `main` branch, the project tests are run.  When it is closed and code is committed to the main branch, the project version is automatically incremented (see below).

### Tests

Blah

### Type Hints

Blah

### Continuos Integration/Continuous Deployment (CI/CD) Workflow

Blah

### Versioning

Semantic versioning (i.e. a scheme that follows a `vMAJOR.MINOR.PATCH` format; see <https://semver.org> for details) is used for this project.  The single point of truth for the current production version is the last git tag on the main branch with a `v[0-9]*` format.

Changes are handled by a GitHub workflow which increments the version and creates a new tag whenever a push occurs to the `main` branch.  This ensures that every commit on the `main` branch is assigned a unique version.  The logic by which it modifies the version is as follows:

1. if the PR message (or one of its commits' messages) contains the text `[version:major]`, then `MAJOR` is incremented;
2. else if the PR message (or one of its commits' messages) contains the text `[version:minor]`, then `MINOR` is incremented;
3. else `PATCH` is incremented.

A `MAJOR` version change should be indicated if the PR introduces a breaking change.  A `MINOR` version change should be indicated if the PR introduces new functionality.

### Git Hooks
This project has been set-up with pre-configured git hooks. They should be used as a means for developers to quickly check that (at least some) of the code standards of the project are being met by commited code.  Ultimately, all standards are actually enforced by the continuous integration pipeline (see below).  Running quick checks (like linting) at the point of commiting code can save time that might otherwise be lost later (for example) at the PR or release stage when testing needs to be rigorous and policy enforcement generally fails slower.  Developers can choose to either:

1. use the git hooks defined by this project (recommended, for the reasons given above; see below for instructions),
2. not to use them, and rely purely on the CI workflow to enforce all project policies, or
3. configure their IDE of choice to manage things, in which case it is up to them to make sure that this aligns with the policies being enforced by CI.

If developers would like to utilise the git hooks provided by this project they just need to run the following command from within the project:
``` console
$ pre-commit
```

#### Maintaining Git Hooks

The git hooks are defined in the `.pre-commit-config.yaml` file.  Specific revisions for many of the tools listed should be managed with Poetry, with syncing managed with the [sync_with_poetry](https://github.com/floatingpurr/sync_with_poetry) hook.  Developers should take care not to use git hooks to *enforce* any project policies.  That should all be done within the continuous integration workflows.  Instead: these should just be quality-of-life checks that fix minor issues or prevent the propagation of quick-and-easy-to-detect problems which would otherwise be caught by the CI later with considerably more latency.  Furthermore, ensure that the checks performed here are consistant between the hooks and the CI.  For example: make sure that any linting/code quality checks are executed with the same tools and options.

### Releases
Releases are generated through the GitHub UI.  A GitHub workflow has been configured to do the following when a new release is produced:

1. Run the tests for the project,
2. Ensure that the project builds,
3. Rebuild the documentation on *Read the Docs*, and
4. Publish a new version of the code on [PyPI](https://pypi.org/).

#### Generating a new release
To generate a new release, do the following:
1. Navigate to the project's GitHub page,
2. Click on `Releases` in the sidebar,
3. Click on `Create a new release` (if this is the first release you have generated) or `Draft release` if this is a subsequent release,
4. Click on `Choose a tag` and select the most recent version listed,
5. Write some text describing the nature of the release to prospective users, and
6. Click `Publish Release`.

### Documentation

Documentation for this project is generated using [Sphinx](https://www.sphinx-doc.org/en/master/) and is hosted on *Read the Docs* for the latest release version.  Sphinx is configured here in the following ways:

1. **Content is managed with Markdown (`.md`) rather than Restructured Text (`.rst`)**

    Developers are mostly spared the pain of direcly editing `.rst` files (the usual way of generating content for Sphinx) in the following ways:
    * default `.rst` files are generated by `sphinx-apidoc` from [Jinja2](https://jinja.palletsprojects.com/en/latest/) templates placed in the `docs/_templates` directory of the project.
    * [MyST-Parser](https://myst-parser.readthedocs.io/en/latest/) is used to source all content from Markdown files.  MyST-Parser also offers [several optional Markdown extensions](https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html) enabling the rendering of richer content (e.g. Latex equations).  Several of these extensions have been enabled by default, but not all.  This can be managed by overriding the behavior of the `conf.py` template (see below) and editing the `myst_enable_extensions` list therein.

2.  **A single point of truth for high-level aspects of the documentation**

    The project `README.md` is utilised, creating a single point of truth for the main high-level aspects of the documentation for both this documentation and all the homepages associated with the services used by this project (see above).

3. **As much content as possible is generated from the code itself**

    By default, the`.rst` templates use the content of the project's `README.md` to create the documentation homepage, followed by the following sections:

    a. _Getting started_, generated from `docs/content/getting_started.md`,

    b. _CLI Documentation_, generated automatically from any *Click*-based CLI applications that are part of the project,

    c. _API Documentation_, generated by `spinx-autodoc` from the docstrings of the project's Python code,

    d. _Notes for Developers_ (i.e. this page), generated from `docs/content/notes_for_developers.md`.

#### Overriding the default behavior of the templates

The behavior of any of the template-generated files can be overridden by placing an alternate version of it in `docs/content`.  This will be copied over top of any template-generated files and then used in their stead.  The easiest way to create such a file (if it doesn't already exist) is to generate the documentation once and then copy the file you wish to override into `docs/content`.  This copy of the file can then be edited.

Some examples of changes you may wish to make:
* new sections can be added to the documentation by overriding `index.rst` and adding a reference to a new file (see below for more details)
* new MyST-Parser extensions can be enabled by overriding `conf.py` and extending the `myst_enable_extensions` list.

#### Generating the Documentation
Documentation can be generated locally by running
``` console
make html
```

from the root directory of the code repository.  This will generate an html version of the documentation in `docs/_build/html`) which can be opened in your browser.  On a Mac (for example), this can be done by running the following:
``` console
$ open docs/_build/html/index.html
```

#### Editing the Documentation

The majority of documentation changes can be managed in one of the following 4 ways:

1. **Edits to `README.md`**:

	Most high-level content should be presented in the `README.md` file.  This content gets used by the project documentation and is shared by the GitHub project page and the PyPI page.

2. **Project Docstrings**:

	Documentation for code changes specifying the codebase's API, implementation details, etc. should be managed directly in the Docstrings of the project's `.py` files.  This content will automatically be harvested by `sphinx-apidoc`.

3. **Existing Markdown files in the `docs` directory**:

	Examine the Markdown files in the `docs/content` directory.  Does the content that you want to add fit naturally within one of those files?  If so: add it there.

4. **Add a new Markdown file to the `docs` directory**:

	Otherwise, create a new `.md` file in the `docs` directory and add it to the list of Markdown files listed in the `docs/content/index.rst` file.  Note that these files will be added to the documentation in the order specified, so place it in that list where you want it to appear in the final documentation.  This new `.md` file should start with a top-level title (marked-up by starting a line with a single `#`; see the top of this file for an example).

#### Adding images, etc.

While not strictly required, it is best practice to place any images, plots, etc. used in the documentation in the `docs/assets` directory.
