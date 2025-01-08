
import { test, expect } from '@playwright/test';

test.describe('https://github.com/about', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('https://github.com/about');
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
        await expect(page).toHaveURL('https://github.com/features/copilot/?utm_source=github&utm_medium=banner&utm_campaign=copilotfree-bannerheader-about');
    });
    

    test('should handle a - Homepage', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Homepage' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Homepage' }).click()
        ]);
        await expect(page).toHaveURL('/');
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
        await expect(page).toHaveURL('/login?return_to=https%3A%2F%2Fgithub.com%2Fabout');
    });
    

    test('should handle a - Sign up', async ({ page }) => {
                await expect(page.getByText('Sign up')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Sign up').click()
        ]);
        await expect(page).toHaveURL('/signup?ref_cta=Sign+up&ref_loc=header+logged+out&ref_page=%2Fabout&source=header');
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
    

    test('should handle a - Blog 

        

        
    ', async ({ page }) => {
                await expect(page.getByText('Blog 

        

        
          Read up on product innovations and updates, company announcements, community spotlights, and more.')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Blog 

        

        
          Read up on product innovations and updates, company announcements, community spotlights, and more.').click()
        ]);
        await expect(page).toHaveURL('https://github.blog');
    });
    

    test('should handle a - Brand assets 

        

     ', async ({ page }) => {
                await expect(page.getByText('Brand assets 

        

        
          Want to use Mona the octocat? Looking for the right way to display the GitHub logo for your latest project? Download the assets and see how and where to use them.')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Brand assets 

        

        
          Want to use Mona the octocat? Looking for the right way to display the GitHub logo for your latest project? Download the assets and see how and where to use them.').click()
        ]);
        await expect(page).toHaveURL('/logos');
    });
    

    test('should handle a - Community stories 

        

', async ({ page }) => {
                await expect(page.getByText('Community stories 

        

        
          Developers are building the future on GitHub every day, explore their stories, celebrate their accomplishments, and find inspiration for your own work.')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Community stories 

        

        
          Developers are building the future on GitHub every day, explore their stories, celebrate their accomplishments, and find inspiration for your own work.').click()
        ]);
        await expect(page).toHaveURL('/readme');
    });
    

    test('should handle a - Customer stories 

        

 ', async ({ page }) => {
                await expect(page.getByText('Customer stories 

        

        
          See how some of the most influential businesses around the world use GitHub to provide the best services, products, and experiences for their customers.')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Customer stories 

        

        
          See how some of the most influential businesses around the world use GitHub to provide the best services, products, and experiences for their customers.').click()
        ]);
        await expect(page).toHaveURL('/customer-stories?type=enterprise');
    });
    

    test('should handle a - Careers 

        

        
 ', async ({ page }) => {
                await expect(page.getByText('Careers 

        

        
          Help us build the home for all developers. We’re a passionate group of people dedicated to software development and collaboration. Come join us!')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Careers 

        

        
          Help us build the home for all developers. We’re a passionate group of people dedicated to software development and collaboration. Come join us!').click()
        ]);
        await expect(page).toHaveURL('https://github.careers');
    });
    

    test('should handle a - Diversity, Inclusions & Belong', async ({ page }) => {
                await expect(page.getByText('Diversity, Inclusions & Belonging 

        

        
          We are dedicated to building a community and team that reflects the world we live in and pushes the boundaries of software innovation. Learn more about our DI&B efforts.')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Diversity, Inclusions & Belonging 

        

        
          We are dedicated to building a community and team that reflects the world we live in and pushes the boundaries of software innovation. Learn more about our DI&B efforts.').click()
        ]);
        await expect(page).toHaveURL('/about/diversity');
    });
    

    test('should handle a - GitHub Status 

        

    ', async ({ page }) => {
                await expect(page.getByText('GitHub Status 

        

        
          We are always monitoring the status of github.com and all its related services. Updates and status interruptions are posted in real-time here.')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub Status 

        

        
          We are always monitoring the status of github.com and all its related services. Updates and status interruptions are posted in real-time here.').click()
        ]);
        await expect(page).toHaveURL('https://www.githubstatus.com');
    });
    

    test('should handle a - Leadership 

        

       ', async ({ page }) => {
                await expect(page.getByText('Leadership 

        

        
          Meet the leadership team guiding us as we continue on this journey building the world’s largest and most advanced software development platform in the world.')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Leadership 

        

        
          Meet the leadership team guiding us as we continue on this journey building the world’s largest and most advanced software development platform in the world.').click()
        ]);
        await expect(page).toHaveURL('/about/leadership');
    });
    

    test('should handle a - Octoverse 

        

        ', async ({ page }) => {
                await expect(page.getByText('Octoverse 

        

        
          Dive into the details with our annual State of the Octoverse report looking at the trends and patterns in the code and communities that build on GitHub.')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Octoverse 

        

        
          Dive into the details with our annual State of the Octoverse report looking at the trends and patterns in the code and communities that build on GitHub.').click()
        ]);
        await expect(page).toHaveURL('https://octoverse.github.com');
    });
    

    test('should handle a - Policy 

        

        
  ', async ({ page }) => {
                await expect(page.getByText('Policy 

        

        
          We’re focused on fighting for developer rights by shaping the policies that promote their interests and the future of software. Learn more about our policy efforts.')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Policy 

        

        
          We’re focused on fighting for developer rights by shaping the policies that promote their interests and the future of software. Learn more about our policy efforts.').click()
        ]);
        await expect(page).toHaveURL('/about/developer-policy');
    });
    

    test('should handle a - Press 

        

        
   ', async ({ page }) => {
                await expect(page.getByText('Press 

        

        
          Explore the latest press stories on our company, products, and global community.')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Press 

        

        
          Explore the latest press stories on our company, products, and global community.').click()
        ]);
        await expect(page).toHaveURL('/about/press');
    });
    

    test('should handle a - Social Impact 

        

    ', async ({ page }) => {
                await expect(page.getByText('Social Impact 

        

        
          Learn about how GitHub’s people, products, and platform are creating positive and lasting change around the world.')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Social Impact 

        

        
          Learn about how GitHub’s people, products, and platform are creating positive and lasting change around the world.').click()
        ]);
        await expect(page).toHaveURL('https://socialimpact.github.com');
    });
    

    test('should handle a - Go to GitHub homepage', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Go to GitHub homepage' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Go to GitHub homepage' }).click()
        ]);
        await expect(page).toHaveURL('/');
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
        await expect(page).toHaveURL('/features');
    });
    

    test('should handle a - Enterprise', async ({ page }) => {
                await expect(page.getByText('Enterprise')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Enterprise').click()
        ]);
        await expect(page).toHaveURL('/enterprise');
    });
    

    test('should handle a - Copilot', async ({ page }) => {
                await expect(page.getByText('Copilot')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Copilot').click()
        ]);
        await expect(page).toHaveURL('/features/copilot');
    });
    

    test('should handle a - Security', async ({ page }) => {
                await expect(page.getByText('Security')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Security').click()
        ]);
        await expect(page).toHaveURL('/security');
    });
    

    test('should handle a - Pricing', async ({ page }) => {
                await expect(page.getByText('Pricing')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Pricing').click()
        ]);
        await expect(page).toHaveURL('/pricing');
    });
    

    test('should handle a - Team', async ({ page }) => {
                await expect(page.getByText('Team')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Team').click()
        ]);
        await expect(page).toHaveURL('/team');
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
        await expect(page).toHaveURL('/enterprise/premium-support');
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
        await expect(page).toHaveURL('/customer-stories?type=enterprise');
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
        await expect(page).toHaveURL('/readme');
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
        await expect(page).toHaveURL('/newsroom');
    });
    

    test('should handle a - Inclusion', async ({ page }) => {
                await expect(page.getByText('Inclusion')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Inclusion').click()
        ]);
        await expect(page).toHaveURL('/about/diversity');
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
        await expect(page).toHaveURL('/sitemap');
    });
    

    test('should handle a - What is Git?', async ({ page }) => {
                await expect(page.getByText('What is Git?')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('What is Git?').click()
        ]);
        await expect(page).toHaveURL('/git-guides');
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
    

    test('should handle global-banner - Announcement', async ({ page }) => {
                await expect(page.getByRole('alert', { name: 'Announcement' })).toBeVisible();
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
        await expect(page).toHaveURL('/');
    });
    

    test('should handle nav - Global', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Global' })).toBeVisible();
    });
    

    test('should handle button - Search or jump to…', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Search or jump to…' })).toBeVisible();
        await page.getByRole('generic', { name: 'Search or jump to…' }).click();
    });
    

    test('should handle a - Go to GitHub homepage', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Go to GitHub homepage' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Go to GitHub homepage' }).click()
        ]);
        await expect(page).toHaveURL('/');
    });
    

    test('should handle nav - Legal and Resource Links', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Legal and Resource Links' })).toBeVisible();
    });
    

    test('should handle nav - GitHubs Social Media Links', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'GitHub's Social Media Links' })).toBeVisible();
    });
    
});
