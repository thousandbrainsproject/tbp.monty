---
title: Pull Requests
---
Monty uses Github Pull Requests to integrate code changes.

# Before making a pull request

## Contributor License Agreement

Before we can accept your contribution, you must sign the Contributor License Agreement (CLA). You can [view and sign the CLA now](https://na4.documents.adobe.com/public/esignWidget?wid=CBFCIBAA3AAABLblqZhA-C5ccSQcDGY-PiamH4HnZdj5p2I1oDc8FiBJ_23pReFeauFhfcIkC1XfzxC2qnBQ*) or wait until you submit your Pull Request.

See the [Contributor License Agreement](pull-requests/contributor-license-agreement.md) page for more on the CLA.

## First-time Contributor

Before submitting a Pull Request, you should set up your development environment to work with Monty. See the development [Getting Started](../how-to-use-monty/getting-started.md) guide for more information.

# Overall Workflow

1. [Identify an issue to work on](ways-to-contribute-to-code/identify-an-issue-to-work-on.md).
2. Ensure your fork has the latest upstream `main` branch changes (if you don't have a fork of the Monty repository or aren't sure, see the development [Getting Started](../how-to-use-monty/getting-started.md) guide):
   ```shell
   git checkout main
   git pull --rebase upstream main
   ```
3. Create a new branch on your fork to work on the issue:
   ```shell
   git checkout -b <my_branch_name>
   ```
4. Implement your changes. Keep in mind any tests or benchmarks that you may need to add or update.
5. If you've added/deleted/modified code, test your changes locally via:
   ```shell
   pytest
   ```
6. Push your changes to your branch on your fork:
   ```shell
   git push
   ```
7. [Create a new Github Pull Request from your fork to the official Monty repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).
8. Respond to and address any comments on your Pull Request. See [Pull Request Flow](pull-requests/pull-request-flow.md) for what to expect.
9. Once your Pull Request is approved, it will be merged by one of the Maintainers. Thank you for contributing! 🥳🎉🎊

## Additional Recommendations for Code Changes

- It is recommended to **add unit tests for any new feature** you implement. This makes sure that your feature continues to function as intended when other people (or you) make future changes to the code. To get a detailed coverage report use `pytest --cov --cov-report html`.
- **Run `pytest` and `ruff check`** to make sure your changes don't break any existing code and adhere to our [style requirements](style-guide.md). If your code doesn't pass these, it can not be merged.
- Make sure that your **code is properly documented**. Please refer to our [Style Guide](style-guide.md) for instructions on how to format your comments.
- If applicable, please also **update or add to the documentation on readme.com**. For instructions on how to do this, see our [guide on contributing documentation](documentation.md).
- **Use callbacks for logging**, and don’t put control logic into logging functions.
- Note that the random seed in Monty is handled using a generator object that is passed
  where needed, i.e. by initializing the random number generator with
  ```
  rng = np.random.RandomState(experiment_args["seed"])
  ```
  This rng is then passed to the various classes, and can be accessed in the sensor
  modules, learning modules, and motor system with self.rng. Thus to use a random
  numpy method, call it with e.g. `self.rng.uniform()` rather than `np.random.uniform()`