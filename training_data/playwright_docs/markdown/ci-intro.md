Source: https://playwright.dev/docs/ci-intro

  * [](/)
  * Getting Started
  * Setting up CI



On this page

# Setting up CI

## Introduction​

Playwright tests can be run on any CI provider. This guide covers one way of running tests on GitHub using GitHub actions. If you would like to learn more, or how to configure other CI providers, check out our detailed [doc on Continuous Integration](/docs/ci).

#### You will learn​

  * [How to set up GitHub Actions](/docs/ci-intro#setting-up-github-actions)
  * [How to view test logs](/docs/ci-intro#viewing-test-logs)
  * [How to view the HTML report](/docs/ci-intro#viewing-the-html-report)
  * [How to view the trace](/docs/ci-intro#viewing-the-trace)
  * [How to publish report on the web](/docs/ci-intro#publishing-report-on-the-web)



## Setting up GitHub Actions​

When [installing Playwright](/docs/intro) using the [VS Code extension](/docs/getting-started-vscode) or with `npm init playwright@latest` you are given the option to add a [GitHub Actions](https://docs.github.com/en/actions) workflow. This creates a `playwright.yml` file inside a `.github/workflows` folder containing everything you need so that your tests run on each push and pull request into the main/master branch. Here's how that file looks:

.github/workflows/playwright.yml
    
    
    name: Playwright Tests  
    on:  
      push:  
        branches: [ main, master ]  
      pull_request:  
        branches: [ main, master ]  
    jobs:  
      test:  
        timeout-minutes: 60  
        runs-on: ubuntu-latest  
        steps:  
        - uses: actions/checkout@v4  
        - uses: actions/setup-node@v4  
          with:  
            node-version: lts/*  
        - name: Install dependencies  
          run: npm ci  
        - name: Install Playwright Browsers  
          run: npx playwright install --with-deps  
        - name: Run Playwright tests  
          run: npx playwright test  
        - uses: actions/upload-artifact@v4  
          if: ${{ !cancelled() }}  
          with:  
            name: playwright-report  
            path: playwright-report/  
            retention-days: 30  
    

The workflow performs these steps:

  1. Clone your repository 2. Install Node.js 3. Install NPM Dependencies 4. Install Playwright Browsers 5. Run Playwright tests 6. Upload HTML report to the GitHub UI



To learn more about this, see ["Understanding GitHub Actions"](https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions).

## Create a Repo and Push to GitHub​

Once you have your GitHub actions workflow setup then all you need to do is [Create a repo on GitHub](https://docs.github.com/en/get-started/quickstart/create-a-repo) or push your code to an existing repository. Follow the instructions on GitHub and don't forget to [initialize a git repository](https://github.com/git-guides/git-init) using the `git init` command so you can [add](https://github.com/git-guides/git-add), [commit](https://github.com/git-guides/git-commit) and [push](https://github.com/git-guides/git-push) your code.

###### 

## Opening the Workflows​

Click on the **Actions** tab to see the workflows. Here you will see if your tests have passed or failed.

###### ​

## Viewing Test Logs​

Clicking on the workflow run will show you the all the actions that GitHub performed and clicking on **Run Playwright tests** will show the error messages, what was expected and what was received as well as the call log.

###### ​

## HTML Report​

The HTML Report shows you a full report of your tests. You can filter the report by browsers, passed tests, failed tests, skipped tests and flaky tests.

### Downloading the HTML Report​

In the Artifacts section click on the **playwright-report** to download your report in the format of a zip file.

### Viewing the HTML Report​

Locally opening the report will not work as expected as you need a web server in order for everything to work correctly. First, extract the zip, preferably in a folder that already has Playwright installed. Using the command line change into the directory where the report is and use `npx playwright show-report` followed by the name of the extracted folder. This will serve up the report and enable you to view it in your browser.
    
    
    npx playwright show-report name-of-my-extracted-playwright-report  
    

To learn more about reports check out our detailed guide on [HTML Reporter](/docs/test-reporters#html-reporter)

## Viewing the Trace​

Once you have served the report using `npx playwright show-report`, click on the trace icon next to the test's file name as seen in the image above. You can then view the trace of your tests and inspect each action to try to find out why the tests are failing.

## Publishing report on the web​

Downloading the HTML report as a zip file is not very convenient. However, we can utilize Azure Storage's static websites hosting capabilities to easily and efficiently serve HTML reports on the Internet, requiring minimal configuration.

  1. Create an [Azure Storage account](https://learn.microsoft.com/en-us/azure/storage/common/storage-account-create).

  2. Enable [Static website hosting](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-static-website-how-to#enable-static-website-hosting) for the storage account.

  3. Create a Service Principal in Azure and grant it access to Azure Blob storage. Upon successful execution, the command will display the credentials which will be used in the next step.
    
        az ad sp create-for-rbac --name "github-actions" --role "Storage Blob Data Contributor" --scopes /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP_NAME>/providers/Microsoft.Storage/storageAccounts/<STORAGE_ACCOUNT_NAME>  
    

  4. Use the credentials from the previous step to set up encrypted secrets in your GitHub repository. Go to your repository's settings, under [GitHub Actions secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-a-repository), and add the following secrets:

     * `AZCOPY_SPA_APPLICATION_ID`
     * `AZCOPY_SPA_CLIENT_SECRET`
     * `AZCOPY_TENANT_ID`

For a detailed guide on how to authorize a service principal using a client secret, refer to [this Microsoft documentation](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-authorize-azure-active-directory#authorize-a-service-principal-by-using-a-client-secret-1).

  5. Add a step that uploads the HTML report to Azure Storage.

.github/workflows/playwright.yml
    
        ...  
        - name: Upload HTML report to Azure  
          shell: bash  
          run: |  
            REPORT_DIR='run-${{ github.run_id }}-${{ github.run_attempt }}'  
            azcopy cp --recursive "./playwright-report/*" "https://<STORAGE_ACCOUNT_NAME>.blob.core.windows.net/\$web/$REPORT_DIR"  
            echo "::notice title=HTML report url::https://<STORAGE_ACCOUNT_NAME>.z1.web.core.windows.net/$REPORT_DIR/index.html"  
          env:  
            AZCOPY_AUTO_LOGIN_TYPE: SPN  
            AZCOPY_SPA_APPLICATION_ID: '${{ secrets.AZCOPY_SPA_APPLICATION_ID }}'  
            AZCOPY_SPA_CLIENT_SECRET: '${{ secrets.AZCOPY_SPA_CLIENT_SECRET }}'  
            AZCOPY_TENANT_ID: '${{ secrets.AZCOPY_TENANT_ID }}'  
    




The contents of the `$web` storage container can be accessed from a browser by using the [public URL](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blob-static-website-how-to?tabs=azure-portal#portal-find-url) of the website.

note

This step will not work for pull requests created from a forked repository because such workflow [doesn't have access to the secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets#using-encrypted-secrets-in-a-workflow).

## What's Next​

  * [Learn how to use Locators](/docs/locators)
  * [Learn how to perform Actions](/docs/input)
  * [Learn how to write Assertions](/docs/test-assertions)
  * [Learn more about the Trace Viewer](/docs/trace-viewer)
  * [Learn more ways of running tests on GitHub Actions](/docs/ci#github-actions)
  * [Learn more about running tests on other CI providers](/docs/ci)


