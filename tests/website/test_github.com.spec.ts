
import { test, expect } from '@playwright/test';

test.describe('https://github.com', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('https://github.com');
    });
    
    
    test('should handle button - Close', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Close' })).toBeVisible();
        await page.getByRole('generic', { name: 'Close' }).click();
    });
    

    test('should handle button - Product', async ({ page }) => {
                await expect(page.getByText('Product')).toBeVisible();
        await page.getByText('Product').click();
    });
    

    test('should handle button - Solutions', async ({ page }) => {
                await expect(page.getByText('Solutions')).toBeVisible();
        await page.getByText('Solutions').click();
    });
    

    test('should handle button - Resources', async ({ page }) => {
                await expect(page.getByText('Resources')).toBeVisible();
        await page.getByText('Resources').click();
    });
    

    test('should handle button - Open Source', async ({ page }) => {
                await expect(page.getByText('Open Source')).toBeVisible();
        await page.getByText('Open Source').click();
    });
    

    test('should handle button - Enterprise', async ({ page }) => {
                await expect(page.getByText('Enterprise')).toBeVisible();
        await page.getByText('Enterprise').click();
    });
    

    test('should handle button - Search or jump to…', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Search or jump to…' })).toBeVisible();
        await page.getByRole('generic', { name: 'Search or jump to…' }).click();
    });
    

    test('should handle button - Sign up for GitHub', async ({ page }) => {
                await expect(page.getByText('Sign up for GitHub')).toBeVisible();
        await page.getByText('Sign up for GitHub').click();
    });
    

    test('should handle button - Pause demo', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Pause demo' })).toBeVisible();
        await page.getByRole('generic', { name: 'Pause demo' }).click();
    });
    

    test('should handle button - Code', async ({ page }) => {
                await expect(page.getByText('Code')).toBeVisible();
        await page.getByText('Code').click();
    });
    

    test('should handle button - Plan', async ({ page }) => {
                await expect(page.getByText('Plan')).toBeVisible();
        await page.getByText('Plan').click();
    });
    

    test('should handle button - Collaborate', async ({ page }) => {
                await expect(page.getByText('Collaborate')).toBeVisible();
        await page.getByText('Collaborate').click();
    });
    

    test('should handle button - Automate', async ({ page }) => {
                await expect(page.getByText('Automate')).toBeVisible();
        await page.getByText('Automate').click();
    });
    

    test('should handle button - Secure', async ({ page }) => {
                await expect(page.getByText('Secure')).toBeVisible();
        await page.getByText('Secure').click();
    });
    

    test('should handle button - Pause video', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Pause video' })).toBeVisible();
        await page.getByRole('generic', { name: 'Pause video' }).click();
    });
    

    test('should handle button - 1', async ({ page }) => {
                await expect(page.getByText('1')).toBeVisible();
        await page.getByText('1').click();
    });
    

    test('should handle button - Automate any workflow', async ({ page }) => {
                await expect(page.getByText('Automate any workflow')).toBeVisible();
        await page.getByText('Automate any workflow').click();
    });
    

    test('should handle button - Get up and running in seconds', async ({ page }) => {
                await expect(page.getByText('Get up and running in seconds')).toBeVisible();
        await page.getByText('Get up and running in seconds').click();
    });
    

    test('should handle button - Build on the go', async ({ page }) => {
                await expect(page.getByText('Build on the go')).toBeVisible();
        await page.getByText('Build on the go').click();
    });
    

    test('should handle button - Integrate the tools you love', async ({ page }) => {
                await expect(page.getByText('Integrate the tools you love')).toBeVisible();
        await page.getByText('Integrate the tools you love').click();
    });
    

    test('should handle button - 2', async ({ page }) => {
                await expect(page.getByText('2')).toBeVisible();
        await page.getByText('2').click();
    });
    

    test('should handle button - Keep track of your tasks', async ({ page }) => {
                await expect(page.getByText('Keep track of your tasks')).toBeVisible();
        await page.getByText('Keep track of your tasks').click();
    });
    

    test('should handle button - Share ideas and ask questions', async ({ page }) => {
                await expect(page.getByText('Share ideas and ask questions')).toBeVisible();
        await page.getByText('Share ideas and ask questions').click();
    });
    

    test('should handle button - Review code changes together', async ({ page }) => {
                await expect(page.getByText('Review code changes together')).toBeVisible();
        await page.getByText('Review code changes together').click();
    });
    

    test('should handle button - Fund open source projects', async ({ page }) => {
                await expect(page.getByText('Fund open source projects')).toBeVisible();
        await page.getByText('Fund open source projects').click();
    });
    

    test('should handle button - By industry', async ({ page }) => {
                await expect(page.getByText('By industry')).toBeVisible();
        await page.getByText('By industry').click();
    });
    

    test('should handle button - By size', async ({ page }) => {
                await expect(page.getByText('By size')).toBeVisible();
        await page.getByText('By size').click();
    });
    

    test('should handle button - By use case', async ({ page }) => {
                await expect(page.getByText('By use case')).toBeVisible();
        await page.getByText('By use case').click();
    });
    

    test('should handle button - Sign up for GitHub', async ({ page }) => {
                await expect(page.getByText('Sign up for GitHub')).toBeVisible();
        await page.getByText('Sign up for GitHub').click();
    });
    

    test('should handle button - Manage cookies', async ({ page }) => {
                await expect(page.getByText('Manage cookies')).toBeVisible();
        await page.getByText('Manage cookies').click();
    });
    

    test('should handle button - Do not share my personal infor', async ({ page }) => {
                await expect(page.getByText('Do not share my personal information')).toBeVisible();
        await page.getByText('Do not share my personal information').click();
    });
    

    test('should handle a - Skip to content', async ({ page }) => {
                await expect(page.getByText('Skip to content')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Skip to content').click()
        ]);
        await expect(page).toHaveURL('#start-of-content');
    });
    

    test('should handle a - Learn more', async ({ page }) => {
                await expect(page.getByText('Learn more')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Learn more').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/copilot/?utm_source=github&utm_medium=banner&utm_campaign=copilotfree-bannerheader');
    });
    

    test('should handle a - Homepage', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Homepage' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Homepage' }).click()
        ]);
        await expect(page).toHaveURL('https://github.com/');
    });
    

    test('should handle a - Pricing', async ({ page }) => {
                await expect(page.getByText('Pricing')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Pricing').click()
        ]);
        await expect(page).toHaveURL('https://github.com/pricing');
    });
    

    test('should handle a - Sign in', async ({ page }) => {
                await expect(page.getByText('Sign in')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Sign in').click()
        ]);
        await expect(page).toHaveURL('https://github.com/login');
    });
    

    test('should handle a - Sign up', async ({ page }) => {
                await expect(page.getByText('Sign up')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Sign up').click()
        ]);
        await expect(page).toHaveURL('https://github.com/signup?ref_cta=Sign+up&ref_loc=header+logged+out&ref_page=%2F&source=header-home');
    });
    

    test('should handle a - Try GitHub Copilot', async ({ page }) => {
                await expect(page.getByText('Try GitHub Copilot')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Try GitHub Copilot').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/copilot');
    });
    

    test('should handle a - Survey: The AI wave continues ', async ({ page }) => {
                await expect(page.getByText('Survey: The AI wave continues to grow on software development teams, 2024.')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Survey: The AI wave continues to grow on software development teams, 2024.').click()
        ]);
        await expect(page).toHaveURL('https://github.blog/news-insights/research/survey-ai-wave-grows/');
    });
    

    test('should handle a - Explore GitHub Copilot', async ({ page }) => {
                await expect(page.getByText('Explore GitHub Copilot')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Explore GitHub Copilot').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/copilot');
    });
    

    test('should handle a - Read customer story', async ({ page }) => {
                await expect(page.getByText('Read customer story')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Read customer story').click()
        ]);
        await expect(page).toHaveURL('https://github.com/customer-stories/duolingo');
    });
    

    test('should handle a - Read report', async ({ page }) => {
                await expect(page.getByText('Read report')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Read report').click()
        ]);
        await expect(page).toHaveURL('https://www.gartner.com/doc/reprints?id=1-2IKO4MPE&ct=240819&st=sb');
    });
    

    test('should handle a - Discover GitHub Actions', async ({ page }) => {
                await expect(page.getByText('Discover GitHub Actions')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Discover GitHub Actions').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/actions');
    });
    

    test('should handle a - Check out GitHub Codespaces', async ({ page }) => {
                await expect(page.getByText('Check out GitHub Codespaces')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Check out GitHub Codespaces').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/codespaces');
    });
    

    test('should handle a - Download GitHub Mobile', async ({ page }) => {
                await expect(page.getByText('Download GitHub Mobile')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Download GitHub Mobile').click()
        ]);
        await expect(page).toHaveURL('https://github.com/mobile');
    });
    

    test('should handle a - Visit GitHub Marketplace', async ({ page }) => {
                await expect(page.getByText('Visit GitHub Marketplace')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Visit GitHub Marketplace').click()
        ]);
        await expect(page).toHaveURL('https://github.com/marketplace');
    });
    

    test('should handle a - Explore GitHub Advanced Securi', async ({ page }) => {
                await expect(page.getByText('Explore GitHub Advanced Security')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Explore GitHub Advanced Security').click()
        ]);
        await expect(page).toHaveURL('https://github.com/enterprise/advanced-security');
    });
    

    test('should handle a - Discover security campaigns', async ({ page }) => {
                await expect(page.getByText('Discover security campaigns')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Discover security campaigns').click()
        ]);
        await expect(page).toHaveURL('https://github.com/enterprise/advanced-security');
    });
    

    test('should handle a - Learn about Dependabot', async ({ page }) => {
                await expect(page.getByText('Learn about Dependabot')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Learn about Dependabot').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/security/software-supply-chain');
    });
    

    test('should handle a - Read about secret scanning', async ({ page }) => {
                await expect(page.getByText('Read about secret scanning')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Read about secret scanning').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/security/code');
    });
    

    test('should handle a - of alert types in all supporte', async ({ page }) => {
                await expect(page.getByText('of alert types in all supported languages with Copilot Autofix')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('of alert types in all supported languages with Copilot Autofix').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com/en/code-security/code-scanning/managing-your-code-scanning-configuration/codeql-query-suites');
    });
    

    test('should handle a - Jump into GitHub Projects', async ({ page }) => {
                await expect(page.getByText('Jump into GitHub Projects')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Jump into GitHub Projects').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/issues');
    });
    

    test('should handle a - Explore GitHub Issues', async ({ page }) => {
                await expect(page.getByText('Explore GitHub Issues')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Explore GitHub Issues').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/issues');
    });
    

    test('should handle a - Discover GitHub Discussions', async ({ page }) => {
                await expect(page.getByText('Discover GitHub Discussions')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Discover GitHub Discussions').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/discussions');
    });
    

    test('should handle a - Learn about code review', async ({ page }) => {
                await expect(page.getByText('Learn about code review')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Learn about code review').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/code-review');
    });
    

    test('should handle a - Dive into GitHub Sponsors', async ({ page }) => {
                await expect(page.getByText('Dive into GitHub Sponsors')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Dive into GitHub Sponsors').click()
        ]);
        await expect(page).toHaveURL('https://github.com/sponsors');
    });
    

    test('should handle a - TechnologyFigma streamlines de', async ({ page }) => {
                await expect(page.getByText('TechnologyFigma streamlines development and strengthens securityRead customer story')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('TechnologyFigma streamlines development and strengthens securityRead customer story').click()
        ]);
        await expect(page).toHaveURL('https://github.com/customer-stories/figma');
    });
    

    test('should handle a - AutomotiveMercedes-Benz standa', async ({ page }) => {
                await expect(page.getByText('AutomotiveMercedes-Benz standardizes source code and automates onboardingRead customer story')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('AutomotiveMercedes-Benz standardizes source code and automates onboardingRead customer story').click()
        ]);
        await expect(page).toHaveURL('https://github.com/customer-stories/mercedes-benz');
    });
    

    test('should handle a - Financial servicesMercado Libr', async ({ page }) => {
                await expect(page.getByText('Financial servicesMercado Libre cuts coding time by 50%Read customer story')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Financial servicesMercado Libre cuts coding time by 50%Read customer story').click()
        ]);
        await expect(page).toHaveURL('https://github.com/customer-stories/mercado-libre');
    });
    

    test('should handle a - Explore customer stories', async ({ page }) => {
                await expect(page.getByText('Explore customer stories')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Explore customer stories').click()
        ]);
        await expect(page).toHaveURL('https://github.com/customer-stories');
    });
    

    test('should handle a - View all solutions', async ({ page }) => {
                await expect(page.getByText('View all solutions')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('View all solutions').click()
        ]);
        await expect(page).toHaveURL('https://github.com/solutions');
    });
    

    test('should handle a - Try GitHub Copilot', async ({ page }) => {
                await expect(page.getByText('Try GitHub Copilot')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Try GitHub Copilot').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/copilot');
    });
    

    test('should handle a - Back to top', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Back to top' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Back to top' }).click()
        ]);
        await expect(page).toHaveURL('#hero');
    });
    

    test('should handle a - Go to GitHub homepage', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Go to GitHub homepage' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Go to GitHub homepage' }).click()
        ]);
        await expect(page).toHaveURL('https://github.com/');
    });
    

    test('should handle a - Subscribe', async ({ page }) => {
                await expect(page.getByText('Subscribe')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Subscribe').click()
        ]);
        await expect(page).toHaveURL('https://resources.github.com/newsletter/');
    });
    

    test('should handle a - Features', async ({ page }) => {
                await expect(page.getByText('Features')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Features').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features');
    });
    

    test('should handle a - Enterprise', async ({ page }) => {
                await expect(page.getByText('Enterprise')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Enterprise').click()
        ]);
        await expect(page).toHaveURL('https://github.com/enterprise');
    });
    

    test('should handle a - Copilot', async ({ page }) => {
                await expect(page.getByText('Copilot')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Copilot').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/copilot');
    });
    

    test('should handle a - Security', async ({ page }) => {
                await expect(page.getByText('Security')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Security').click()
        ]);
        await expect(page).toHaveURL('https://github.com/security');
    });
    

    test('should handle a - Pricing', async ({ page }) => {
                await expect(page.getByText('Pricing')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Pricing').click()
        ]);
        await expect(page).toHaveURL('https://github.com/pricing');
    });
    

    test('should handle a - Team', async ({ page }) => {
                await expect(page.getByText('Team')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Team').click()
        ]);
        await expect(page).toHaveURL('https://github.com/team');
    });
    

    test('should handle a - Resources', async ({ page }) => {
                await expect(page.getByText('Resources')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Resources').click()
        ]);
        await expect(page).toHaveURL('https://resources.github.com');
    });
    

    test('should handle a - Roadmap', async ({ page }) => {
                await expect(page.getByText('Roadmap')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Roadmap').click()
        ]);
        await expect(page).toHaveURL('https://github.com/github/roadmap');
    });
    

    test('should handle a - Compare GitHub', async ({ page }) => {
                await expect(page.getByText('Compare GitHub')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Compare GitHub').click()
        ]);
        await expect(page).toHaveURL('https://resources.github.com/devops/tools/compare');
    });
    

    test('should handle a - Developer API', async ({ page }) => {
                await expect(page.getByText('Developer API')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Developer API').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com/get-started/exploring-integrations/about-building-integrations');
    });
    

    test('should handle a - Partners', async ({ page }) => {
                await expect(page.getByText('Partners')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Partners').click()
        ]);
        await expect(page).toHaveURL('https://partner.github.com');
    });
    

    test('should handle a - Education', async ({ page }) => {
                await expect(page.getByText('Education')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Education').click()
        ]);
        await expect(page).toHaveURL('https://github.com/edu');
    });
    

    test('should handle a - GitHub CLI', async ({ page }) => {
                await expect(page.getByText('GitHub CLI')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub CLI').click()
        ]);
        await expect(page).toHaveURL('https://cli.github.com');
    });
    

    test('should handle a - GitHub Desktop', async ({ page }) => {
                await expect(page.getByText('GitHub Desktop')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub Desktop').click()
        ]);
        await expect(page).toHaveURL('https://desktop.github.com');
    });
    

    test('should handle a - GitHub Mobile', async ({ page }) => {
                await expect(page.getByText('GitHub Mobile')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub Mobile').click()
        ]);
        await expect(page).toHaveURL('https://github.com/mobile');
    });
    

    test('should handle a - Docs', async ({ page }) => {
                await expect(page.getByText('Docs')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Docs').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com');
    });
    

    test('should handle a - Community Forum', async ({ page }) => {
                await expect(page.getByText('Community Forum')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Community Forum').click()
        ]);
        await expect(page).toHaveURL('https://github.community');
    });
    

    test('should handle a - Professional Services', async ({ page }) => {
                await expect(page.getByText('Professional Services')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Professional Services').click()
        ]);
        await expect(page).toHaveURL('https://services.github.com');
    });
    

    test('should handle a - Premium Support', async ({ page }) => {
                await expect(page.getByText('Premium Support')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Premium Support').click()
        ]);
        await expect(page).toHaveURL('https://github.com/enterprise/premium-support');
    });
    

    test('should handle a - Skills', async ({ page }) => {
                await expect(page.getByText('Skills')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Skills').click()
        ]);
        await expect(page).toHaveURL('https://skills.github.com');
    });
    

    test('should handle a - Status', async ({ page }) => {
                await expect(page.getByText('Status')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Status').click()
        ]);
        await expect(page).toHaveURL('https://www.githubstatus.com');
    });
    

    test('should handle a - Contact GitHub', async ({ page }) => {
                await expect(page.getByText('Contact GitHub')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Contact GitHub').click()
        ]);
        await expect(page).toHaveURL('https://support.github.com?tags=dotcom-footer');
    });
    

    test('should handle a - About', async ({ page }) => {
                await expect(page.getByText('About')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('About').click()
        ]);
        await expect(page).toHaveURL('https://github.com/about');
    });
    

    test('should handle a - Customer stories', async ({ page }) => {
                await expect(page.getByText('Customer stories')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Customer stories').click()
        ]);
        await expect(page).toHaveURL('https://github.com/customer-stories?type=enterprise');
    });
    

    test('should handle a - Blog', async ({ page }) => {
                await expect(page.getByText('Blog')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Blog').click()
        ]);
        await expect(page).toHaveURL('https://github.blog');
    });
    

    test('should handle a - The ReadME Project', async ({ page }) => {
                await expect(page.getByText('The ReadME Project')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('The ReadME Project').click()
        ]);
        await expect(page).toHaveURL('https://github.com/readme');
    });
    

    test('should handle a - Careers', async ({ page }) => {
                await expect(page.getByText('Careers')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Careers').click()
        ]);
        await expect(page).toHaveURL('https://github.careers');
    });
    

    test('should handle a - Newsroom', async ({ page }) => {
                await expect(page.getByText('Newsroom')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Newsroom').click()
        ]);
        await expect(page).toHaveURL('https://github.com/newsroom');
    });
    

    test('should handle a - Inclusion', async ({ page }) => {
                await expect(page.getByText('Inclusion')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Inclusion').click()
        ]);
        await expect(page).toHaveURL('https://github.com/about/diversity');
    });
    

    test('should handle a - Social Impact', async ({ page }) => {
                await expect(page.getByText('Social Impact')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Social Impact').click()
        ]);
        await expect(page).toHaveURL('https://socialimpact.github.com');
    });
    

    test('should handle a - Shop', async ({ page }) => {
                await expect(page.getByText('Shop')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Shop').click()
        ]);
        await expect(page).toHaveURL('https://shop.github.com');
    });
    

    test('should handle a - Terms', async ({ page }) => {
                await expect(page.getByText('Terms')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Terms').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com/site-policy/github-terms/github-terms-of-service');
    });
    

    test('should handle a - Privacy', async ({ page }) => {
                await expect(page.getByText('Privacy')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Privacy').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com/site-policy/privacy-policies/github-privacy-statement');
    });
    

    test('should handle a - Updated 02/2024', async ({ page }) => {
                await expect(page.getByText('Updated 02/2024')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Updated 02/2024').click()
        ]);
        await expect(page).toHaveURL('https://github.com/github/site-policy/pull/582');
    });
    

    test('should handle a - Sitemap', async ({ page }) => {
                await expect(page.getByText('Sitemap')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Sitemap').click()
        ]);
        await expect(page).toHaveURL('https://github.com/sitemap');
    });
    

    test('should handle a - What is Git?', async ({ page }) => {
                await expect(page.getByText('What is Git?')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('What is Git?').click()
        ]);
        await expect(page).toHaveURL('https://github.com/git-guides');
    });
    

    test('should handle a - GitHub on LinkedIn', async ({ page }) => {
                await expect(page.getByText('GitHub on LinkedIn')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub on LinkedIn').click()
        ]);
        await expect(page).toHaveURL('https://www.linkedin.com/company/github');
    });
    

    test('should handle a - Instagram

        GitHub on I', async ({ page }) => {
                await expect(page.getByText('Instagram

        GitHub on Instagram')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Instagram

        GitHub on Instagram').click()
        ]);
        await expect(page).toHaveURL('https://www.instagram.com/github');
    });
    

    test('should handle a - GitHub on YouTube', async ({ page }) => {
                await expect(page.getByText('GitHub on YouTube')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub on YouTube').click()
        ]);
        await expect(page).toHaveURL('https://www.youtube.com/github');
    });
    

    test('should handle a - GitHub on X', async ({ page }) => {
                await expect(page.getByText('GitHub on X')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub on X').click()
        ]);
        await expect(page).toHaveURL('https://x.com/github');
    });
    

    test('should handle a - TikTok

        GitHub on TikT', async ({ page }) => {
                await expect(page.getByText('TikTok

        GitHub on TikTok')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('TikTok

        GitHub on TikTok').click()
        ]);
        await expect(page).toHaveURL('https://www.tiktok.com/@github');
    });
    

    test('should handle a - Twitch

        GitHub on Twit', async ({ page }) => {
                await expect(page.getByText('Twitch

        GitHub on Twitch')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Twitch

        GitHub on Twitch').click()
        ]);
        await expect(page).toHaveURL('https://www.twitch.tv/github');
    });
    

    test('should handle a - GitHub’s organization on GitHu', async ({ page }) => {
                await expect(page.getByText('GitHub’s organization on GitHub')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub’s organization on GitHub').click()
        ]);
        await expect(page).toHaveURL('https://github.com/github');
    });
    

    test('should handle input - you@domain.com', async ({ page }) => {
                await expect(page.locator('#hero_user_email')).toBeVisible();
        await page.locator('#hero_user_email').click();
        await page.locator('#hero_user_email').fill('test@example.com');
        await expect(page.locator('#hero_user_email')).toHaveValue('test@example.com');
    });
    

    test('should handle input - you@domain.com', async ({ page }) => {
                await expect(page.locator('#bottom_cta_section_user_email')).toBeVisible();
        await page.locator('#bottom_cta_section_user_email').click();
        await page.locator('#bottom_cta_section_user_email').fill('test@example.com');
        await expect(page.locator('#bottom_cta_section_user_email')).toHaveValue('test@example.com');
    });
    

    test('should handle button - Close', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Close' })).toBeVisible();
        await page.getByRole('generic', { name: 'Close' }).click();
    });
    

    test('should handle a - Homepage', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Homepage' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Homepage' }).click()
        ]);
        await expect(page).toHaveURL('https://github.com/');
    });
    

    test('should handle button - Search or jump to…', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Search or jump to…' })).toBeVisible();
        await page.getByRole('generic', { name: 'Search or jump to…' }).click();
    });
    

    test('should handle button - Pause demo', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Pause demo' })).toBeVisible();
        await page.getByRole('generic', { name: 'Pause demo' }).click();
    });
    

    test('should handle button - Pause video', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Pause video' })).toBeVisible();
        await page.getByRole('generic', { name: 'Pause video' }).click();
    });
    

    test('should handle a - Back to top', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Back to top' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Back to top' }).click()
        ]);
        await expect(page).toHaveURL('#hero');
    });
    

    test('should handle a - Go to GitHub homepage', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Go to GitHub homepage' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Go to GitHub homepage' }).click()
        ]);
        await expect(page).toHaveURL('https://github.com/');
    });
    
});
