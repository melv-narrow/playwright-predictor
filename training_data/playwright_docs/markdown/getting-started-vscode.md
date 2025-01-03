Source: https://playwright.dev/docs/getting-started-vscode

  * [](/)
  * Getting started - VS Code



On this page

# Getting started - VS Code

## Introduction​

Playwright Test was created specifically to accommodate the needs of end-to-end testing. Playwright supports all modern rendering engines including Chromium, WebKit, and Firefox. Test on Windows, Linux, and macOS, locally or on CI, headless or headed with native mobile emulation of Google Chrome for Android and Mobile Safari.

Get started by installing Playwright and generating a test to see it in action. Alternatively you can also get started and run your tests using the [CLI](/docs/intro).

## Installation​

Playwright has a VS Code extension which is available when testing with Node.js. Install [it from the VS Code marketplace](https://marketplace.visualstudio.com/items?itemName=ms-playwright.playwright) or from the extensions tab in VS Code.

Once installed, open the command panel and type:
    
    
    Install Playwright  
    

Select **Test: Install Playwright** and Choose the browsers you would like to run your tests on. These can be later configured in the [playwright.config](/docs/test-configuration) file. You can also choose if you would like to have a GitHub Actions setup to [run your tests on CI](/docs/ci-intro).

### Opening the testing sidebar​

The testing sidebar can be opened by clicking on the testing icon in the activity bar. This will give you access to the test explorer, which will show you all the tests in your project as well as the Playwright sidebar which includes projects, settings, tools and setup.

## Running tests​

You can run a single test by clicking the green triangle next to your test block to run your test. Playwright will run through each line of the test and when it finishes you will see a green tick next to your test block as well as the time it took to run the test.

### Run tests and show browsers​

You can also run your tests and show the browsers by selecting the option **Show Browsers** in the testing sidebar. Then when you click the green triangle to run your test the browser will open and you will visually see it run through your test. Leave this selected if you want browsers open for all your tests or uncheck it if you prefer your tests to run in headless mode with no browser open.

Use the **Close all browsers** button to close all browsers.

### View and run all tests​

View all tests in the testing sidebar and extend the tests by clicking on each test. Tests that have not been run will not have the green check next to them. Run all tests by clicking on the white triangle as you hover over the tests in the testing sidebar.

### Running tests on multiple browsers​

The first section in the Playwright sidebar is the projects section. Here you can see all your projects as defined in your Playwright config file. The default config when installing Playwright gives you 3 projects, Chromium, Firefox and WebKit. The first project is selected by default.

To run tests on multiple projects, select each project by checking the checkboxes next to the project name. Then when you run your tests from the sidebar or by pressing the play button next to the test name, the tests will run on all the selected projects.

You can also individually run a test on a specific project by clicking the grey play button next to the project name of the test.

### Run tests with trace viewer​

For a better developer experience you can run your tests with the **Show Trace Viewer** option.

This will open up a full trace of your test where you can step through each action of your tests, explore the timeline, source code and more.

To learn more about the trace viewer see our [Trace Viewer guide](/docs/trace-viewer).

## Debugging tests​

With the VS Code extension you can debug your tests right in VS Code see error messages, create breakpoints and live debug your tests.

### Error messages​

If your test fails VS Code will show you error messages right in the editor showing what was expected, what was received as well as a complete call log.

### Live debugging​

You can debug your test live in VS Code. After running a test with the `Show Browser` option checked, click on any of the locators in VS Code and it will be highlighted in the Browser window. Playwright will highlight it if it exists and show you if there is more than one result

You can also edit the locators in VS Code and Playwright will show you the changes live in the browser window.

### Run in debug mode​

To set a breakpoint click next to the line number where you want the breakpoint to be until a red dot appears. Run the tests in debug mode by right clicking on the line next to the test you want to run.

A browser window will open and the test will run and pause at where the breakpoint is set. You can step through the tests, pause the test and rerun the tests from the menu in VS Code.

To learn more about debugging, see [Debugging in Visual Studio Code](https://code.visualstudio.com/docs/editor/debugging).

### Debug with trace viewer​

For a better developer experience you can debug your tests with the **Show Trace Viewer** option.

This will open up a full trace of your test where you can step through each action and see what happened before and after the action. You can also inspect the DOM snapshot, see console logs, network requests, the source code and more.

To learn more about the trace viewer see our [Trace Viewer guide](/docs/trace-viewer).

## Generating tests​

CodeGen will auto generate your tests for you as you perform actions in the browser and is a great way to quickly get started. The viewport for the browser window is set to a specific width and height. See the [configuration guide](/docs/test-configuration) to change the viewport or emulate different environments.

### Record a new test​

To record a test click on the **Record new** button from the Testing sidebar. This will create a `test-1.spec.ts` file as well as open up a browser window. In the browser go to the URL you wish to test and start clicking around. Playwright will record your actions and generate the test code directly in VS Code. You can also generate assertions by choosing one of the icons in the toolbar and then clicking on an element on the page to assert against. The following assertions can be generated:

  * `'assert visibility'` to assert that an element is visible
  * `'assert text'` to assert that an element contains specific text
  * `'assert value'` to assert that an element has a specific value



Once you are done recording click the **cancel** button or close the browser window. You can then inspect your `test-1.spec.ts` file and see your generated test.

### Record at cursor​

To record from a specific point in your test file click the **Record at cursor** button from the Testing sidebar. This generates actions into the existing test at the current cursor position. You can run the test, position the cursor at the end of the test and continue generating the test.

### Picking a locator​

Pick a [locator](/docs/locators) and copy it into your test file by clicking the **Pick locator** button form the testing sidebar. Then in the browser click the element you require and it will now show up in the **Pick locator** box in VS Code. Press 'enter' on your keyboard to copy the locator into the clipboard and then paste anywhere in your code. Or press 'escape' if you want to cancel.

Playwright will look at your page and figure out the best locator, prioritizing [role, text and test id locators](/docs/locators). If the generator finds multiple elements matching the locator, it will improve the locator to make it resilient and uniquely identify the target element, so you don't have to worry about failing tests due to locators.

## Project Dependencies​

You can use [project dependencies](/docs/test-projects) to run tests that depend on other tests. This is useful for **setup** tests such as logging in to a website.

### Running setup tests​

To run your setup tests select the **setup** project, as defined in your configuration file, from the project section in the Playwright sidebar. This will give you access to the **setup** tests in the test explorer.

When you run a test that depends on the **setup** tests, the **setup** test will run first. Each time you run the test, the **setup** test will run again.

### Running setup tests only once​

To run the **setup** test only once, deselect it from the projects section in the Playwright sidebar. The **setup** test is now removed from the test explorer. When you run a test that depends on the **setup** test, it will no longer run the **setup** test, making it much faster and therefore a much better developer experience.

## Global Setup​

**Global setup** runs when you execute your first test. It runs only once and is useful for setting up a database or starting a server. You can manually run **global setup** by clicking the `Run global setup` option from the **Setup** section in the Playwright sidebar. **Global teardown** does not run by default; you need to manually initiate it by clicking the `Run global teardown` option.

Global setup will re-run when you debug tests as this ensures an isolated environment and dedicated setup for the test.

## Multiple configurations​

If your project contains more than one playwright configuration file, you can switch between them by first clicking on the gear icon in the top right corner of the Playwright sidebar. This will show you all the configuration files in your project. Select the configuration files you want to use by checking the checkbox next to each one and clicking on the 'ok' button.

You will now have access to all your tests in the test explorer. To run a test click on the grey triangle next to the file or project name.

To run all tests from all configurations click on the grey triangle at the top of the test explorer.

To choose a configuration file to work with simply toggle between them by clicking on the configuration file name in the Playwright sidebar. Now when you use the tools, such as Record a test, it will record a test for the selected configuration file.

You can easily toggle back and forth between configurations by clicking on the configuration file name in the Playwright sidebar.

## What's next​

  * [Write tests using web first assertions, page fixtures and locators](/docs/writing-tests)
  * [Run your tests on CI](/docs/ci-intro)
  * [Learn more about the Trace Viewer](/docs/trace-viewer)


