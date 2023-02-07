# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the owners of this repository before making a change.

Please note we have a code of conduct, please follow it in all your interactions with the project.

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a
   build.
2. Update the README.md with details of changes to the interface, this includes new environment
   variables, exposed ports, useful file locations and container parameters.
3. Increase the version numbers in any examples files and the README.md to the new version that this
   Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/).
4. Regenerate any additional documentation using PDOC (usage details listed below).
5. Document the proposed changes in the CHANGELOG.md file.
6. You may submit your merge request for review and the change will be reviewed.

## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of experience,
nationality, personal appearance, race, religion, or sexual identity and
orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment
include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery and unwelcome sexual attention or
advances
* Trolling, insulting/derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or electronic
  address, without explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable
behavior and are expected to take appropriate and fair corrective action in
response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or
reject comments, commits, code, wiki edits, issues, and other contributions
that are not aligned to this Code of Conduct, or to ban temporarily or
permanently any contributor for other behaviors that they deem inappropriate,
threatening, offensive, or harmful.

### Scope

This Code of Conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community. Examples of
representing a project or community include using an official project e-mail
address, posting via an official social media account, or acting as an appointed
representative at an online or offline event. Representation of a project may be
further defined and clarified by project maintainers.

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported by contacting the project team at [INSERT EMAIL ADDRESS]. All
complaints will be reviewed and investigated and will result in a response that
is deemed necessary and appropriate to the circumstances. The project team is
obligated to maintain confidentiality with regard to the reporter of an incident.
Further details of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in good
faith may face temporary or permanent repercussions as determined by other
members of the project's leadership.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 1.4,
available at [http://contributor-covenant.org/version/1/4][version]

[homepage]: http://contributor-covenant.org
[version]: http://contributor-covenant.org/version/1/4/

## Appendix

### Generating Documentation

This repository follows semi-automatic documentation generation. The following
is an example of how to generate documentation for a single module.

```bash
conda activate tensorflow-caney
pdoc --html tensorflow-caney/raster.py --force
```

### Linting

This project uses flake8 for PREP8 linting and format. Every submodule should include
a test section in the tests directory. Refer to the text directory for more examples.
The Python unittests library is used for these purposes.

### Documenting Methods

The following documentation format should be followed below each method to allow for
explicit semi-automatic documentation generation.

```bash
   """
   Read raster and append data to existing Raster object
   Args:
      filename (str): raster filename to read from
      bands (str list): list of bands to append to object, e.g ['Red']
      chunks_band (int): integer to map object to memory, z
      chunks_x (int): integer to map object to memory, x
      chunks_y (int): integer to map object to memory, y
   Return:
      raster (raster object): raster object to manipulate rasters
   ----------
   Example
   ----------
      raster.readraster(filename, bands)
   """
```

### Format of CHANGELOG

The following describes the format for each CHANGELOG release. If there are no contributions
in any of the sections, they are removed from the description.

```bash
## [0.0.3] - 2020-12-14

### Added
- Short description

### Fixed
- Short description

### Changed
- Short description

### Removed
- Short description

### Approved
Approver Name, Email
```

### Example Using Container in ADAPT

```bash
module load singularity
singularity shell -B $your_mounts --nv tensorflow-caney
```

### Current Workflow

```bash
module load singularity
singularity shell --nv -B /lscratch,/css,/explore/nobackup/projects/ilab,/explore/nobackup/people /explore/nobackup/projects/ilab/containers/tensorflow-caney-2022.12
export PYTHONPATH="/explore/nobackup/people/jacaraba/development/tensorflow-caney:/adapt/nobackup/people/jacaraba/development/vhr-cnn-chm"
```
