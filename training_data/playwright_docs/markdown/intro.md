Source: https://playwright.dev/docs/intro

  * [](/)
  * Getting Started
  * Installation



On this page

# Installation

## Introduction​

Playwright Test was created specifically to accommodate the needs of end-to-end testing. Playwright supports all modern rendering engines including Chromium, WebKit, and Firefox. Test on Windows, Linux, and macOS, locally or on CI, headless or headed with native mobile emulation of Google Chrome for Android and Mobile Safari.

**You will learn**

  * [How to install Playwright](/docs/intro#installing-playwright)
  * [What's Installed](/docs/intro#whats-installed)
  * [How to run the example test](/docs/intro#running-the-example-test)
  * [How to open the HTML test report](/docs/intro#html-test-reports)



## Installing Playwright​

Get started by installing Playwright using npm, yarn or pnpm. Alternatively you can also get started and run your tests using the [VS Code Extension](/docs/getting-started-vscode).

  * npm
  * yarn
  * pnpm


    
    
    npm init playwright@latest  
    
    
    
    yarn create playwright  
    
    
    
    pnpm create playwright  
    

Run the install command and select the following to get started:

  * Choose between TypeScript or JavaScript (default is TypeScript)
  * Name of your Tests folder (default is tests or e2e if you already have a tests folder in your project)
  * Add a GitHub Actions workflow to easily run tests on CI
  * Install Playwright browsers (default is true)



## What's Installed​

Playwright will download the browsers needed as well as create the following files.
    
    
    playwright.config.ts  
    package.json  
    package-lock.json  
    tests/  
      example.spec.ts  
    tests-examples/  
      demo-todo-app.spec.ts  
    

The [playwright.config](/docs/test-configuration) is where you can add configuration for Playwright including modifying which browsers you would like to run Playwright on. If you are running tests inside an already existing project then dependencies will be added directly to your `package.json`.

The `tests` folder contains a basic example test to help you get started with testing. For a more detailed example check out the `tests-examples` folder which contains tests written to test a todo app.

## Running the Example Test​

By default tests will be run on all 3 browsers, chromium, firefox and webkit using 3 workers. This can be configured in the [playwright.config file](/docs/test-configuration). Tests are run in headless mode meaning no browser will open up when running the tests. Results of the tests and test logs will be shown in the terminal.

  * npm
  * yarn
  * pnpm


    
    
    npx playwright test  
    
    
    
    yarn playwright test  
    
    
    
    pnpm exec playwright test  
    

See our doc on [Running Tests](/docs/running-tests) to learn more about running tests in headed mode, running multiple tests, running specific tests etc.

## HTML Test Reports​

After your test completes, an [HTML Reporter](/docs/test-reporters#html-reporter) will be generated, which shows you a full report of your tests allowing you to filter the report by browsers, passed tests, failed tests, skipped tests and flaky tests. You can click on each test and explore the test's errors as well as each step of the test. By default, the HTML report is opened automatically if some of the tests failed.

  * npm
  * yarn
  * pnpm


    
    
    npx playwright show-report  
    
    
    
    yarn playwright show-report  
    
    
    
    pnpm exec playwright show-report  
    

## Running the Example Test in UI Mode​

Run your tests with [UI Mode](/docs/test-ui-mode) for a better developer experience with time travel debugging, watch mode and more.

  * npm
  * yarn
  * pnpm


    
    
    npx playwright test --ui  
    
    
    
    yarn playwright test --ui  
    
    
    
    pnpm exec playwright test --ui  
    

Check out or [detailed guide on UI Mode](/docs/test-ui-mode) to learn more about its features.

## Updating Playwright​

To update Playwright to the latest version run the following command:

  * npm
  * yarn
  * pnpm


    
    
    npm install -D @playwright/test@latest  
    # Also download new browser binaries and their dependencies:  
    npx playwright install --with-deps  
    
    
    
    yarn add --dev @playwright/test@latest  
    # Also download new browser binaries and their dependencies:  
    yarn playwright install --with-deps  
    
    
    
    pnpm install --save-dev @playwright/test@latest  
    # Also download new browser binaries and their dependencies:  
    pnpm exec playwright install --with-deps  
    

You can always check which version of Playwright you have by running the following command:

  * npm
  * yarn
  * pnpm


    
    
    npx playwright --version  
    
    
    
    yarn playwright --version  
    
    
    
    pnpm exec playwright --version  
    

## System requirements​

  * Node.js 18+
  * Windows 10+, Windows Server 2016+ or Windows Subsystem for Linux (WSL).
  * macOS 13 Ventura, or later.
  * Debian 12, Ubuntu 22.04, Ubuntu 24.04, on x86-64 and arm64 architecture.



## What's next​

  * [Write tests using web first assertions, page fixtures and locators](/docs/writing-tests)
  * [Run single test, multiple tests, headed mode](/docs/running-tests)
  * [Generate tests with Codegen](/docs/codegen-intro)
  * [See a trace of your tests](/docs/trace-viewer-intro)


