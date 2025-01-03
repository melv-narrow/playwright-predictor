Source: https://playwright.dev/docs/trace-viewer

  * [](/)
  * Guides
  * Trace viewer



On this page

# Trace viewer

## Introduction​

Playwright Trace Viewer is a GUI tool that helps you explore recorded Playwright traces after the script has run. Traces are a great way for debugging your tests when they fail on CI. You can open traces locally or in your browser on [trace.playwright.dev](https://trace.playwright.dev).

###### 

## Trace Viewer features​

### Actions​

In the Actions tab you can see what locator was used for every action and how long each one took to run. Hover over each action of your test and visually see the change in the DOM snapshot. Go back and forward in time and click an action to inspect and debug. Use the Before and After tabs to visually see what happened before and after the action.

**Selecting each action reveals:**

  * action snapshots
  * action log
  * source code location



### Screenshots​

When tracing with the [screenshots](/docs/api/class-tracing#tracing-start-option-screenshots) option turned on (default), each trace records a screencast and renders it as a film strip. You can hover over the film strip to see a magnified image of for each action and state which helps you easily find the action you want to inspect.

Double click on an action to see the time range for that action. You can use the slider in the timeline to increase the actions selected and these will be shown in the Actions tab and all console logs and network logs will be filtered to only show the logs for the actions selected.

### Snapshots​

When tracing with the [snapshots](/docs/api/class-tracing#tracing-start-option-snapshots) option turned on (default), Playwright captures a set of complete DOM snapshots for each action. Depending on the type of the action, it will capture:

Type| Description  
---|---  
Before| A snapshot at the time action is called.  
Action| A snapshot at the moment of the performed input. This type of snapshot is especially useful when exploring where exactly Playwright clicked.  
After| A snapshot after the action.  
  
Here is what the typical Action snapshot looks like:

Notice how it highlights both, the DOM Node as well as the exact click position.

### Source​

When you click on an action in the sidebar, the line of code for that action is highlighted in the source panel.

### Call​

The call tab shows you information about the action such as the time it took, what locator was used, if in strict mode and what key was used.

### Log​

See a full log of your test to better understand what Playwright is doing behind the scenes such as scrolling into view, waiting for element to be visible, enabled and stable and performing actions such as click, fill, press etc.

### Errors​

If your test fails you will see the error messages for each test in the Errors tab. The timeline will also show a red line highlighting where the error occurred. You can also click on the source tab to see on which line of the source code the error is.

### Console​

See console logs from the browser as well as from your test. Different icons are displayed to show you if the console log came from the browser or from the test file.

Double click on an action from your test in the actions sidebar. This will filter the console to only show the logs that were made during that action. Click the _Show all_ button to see all console logs again.

Use the timeline to filter actions, by clicking a start point and dragging to an ending point. The console tab will also be filtered to only show the logs that were made during the actions selected.

### Network​

The Network tab shows you all the network requests that were made during your test. You can sort by different types of requests, status code, method, request, content type, duration and size. Click on a request to see more information about it such as the request headers, response headers, request body and response body.

Double click on an action from your test in the actions sidebar. This will filter the network requests to only show the requests that were made during that action. Click the _Show all_ button to see all network requests again.

Use the timeline to filter actions, by clicking a start point and dragging to an ending point. The network tab will also be filtered to only show the network requests that were made during the actions selected.

### Metadata​

Next to the Actions tab you will find the Metadata tab which will show you more information on your test such as the Browser, viewport size, test duration and more.

### Attachments​

The "Attachments" tab allows you to explore attachments. If you're doing [visual regression testing](/docs/test-snapshots), you'll be able to compare screenshots by examining the image diff, the actual image and the expected image. When you click on the expected image you can use the slider to slide one image over the other so you can easily see the differences in your screenshots.

## Recording a trace locally​

To record a trace during development mode set the `--trace` flag to `on` when running your tests. You can also use [UI Mode](/docs/test-ui-mode) for a better developer experience.
    
    
    npx playwright test --trace on  
    

You can then open the HTML report and click on the trace icon to open the trace.
    
    
    npx playwright show-report  
    

## Recording a trace on CI​

Traces should be run on continuous integration on the first retry of a failed test by setting the `trace: 'on-first-retry'` option in the test configuration file. This will produce a `trace.zip` file for each test that was retried.

  * Test
  * Library



playwright.config.ts
    
    
    import { defineConfig } from '@playwright/test';  
    export default defineConfig({  
      retries: 1,  
      use: {  
        trace: 'on-first-retry',  
      },  
    });  
    
    
    
    const browser = await chromium.launch();  
    const context = await browser.newContext();  
      
    // Start tracing before creating / navigating a page.  
    await context.tracing.start({ screenshots: true, snapshots: true });  
      
    const page = await context.newPage();  
    await page.goto('https://playwright.dev');  
      
    // Stop tracing and export it into a zip archive.  
    await context.tracing.stop({ path: 'trace.zip' });  
    

Available options to record a trace:

  * `'on-first-retry'` \- Record a trace only when retrying a test for the first time.
  * `'on-all-retries'` \- Record traces for all test retries.
  * `'off'` \- Do not record a trace.
  * `'on'` \- Record a trace for each test. (not recommended as it's performance heavy)
  * `'retain-on-failure'` \- Record a trace for each test, but remove it from successful test runs.



You can also use `trace: 'retain-on-failure'` if you do not enable retries but still want traces for failed tests.

There are more granular options available, see [testOptions.trace](/docs/api/class-testoptions#test-options-trace).

If you are not using Playwright as a Test Runner, use the [browserContext.tracing](/docs/api/class-browsercontext#browser-context-tracing) API instead.

## Opening the trace​

You can open the saved trace using the Playwright CLI or in your browser on [`trace.playwright.dev`](https://trace.playwright.dev). Make sure to add the full path to where your `trace.zip` file is located.
    
    
    npx playwright show-trace path/to/trace.zip  
    

## Using [trace.playwright.dev](https://trace.playwright.dev)​

[trace.playwright.dev](https://trace.playwright.dev) is a statically hosted variant of the Trace Viewer. You can upload trace files using drag and drop.

## Viewing remote traces​

You can open remote traces using its URL. They could be generated on a CI run which makes it easy to view the remote trace without having to manually download the file.
    
    
    npx playwright show-trace https://example.com/trace.zip  
    

You can also pass the URL of your uploaded trace (e.g. inside your CI) from some accessible storage as a parameter. CORS (Cross-Origin Resource Sharing) rules might apply.
    
    
    https://trace.playwright.dev/?trace=https://demo.playwright.dev/reports/todomvc/data/cb0fa77ebd9487a5c899f3ae65a7ffdbac681182.zip  
    
