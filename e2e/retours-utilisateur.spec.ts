import { test, expect } from '@playwright/test'

test('test utilisateur', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Créer une session' }).click()
  await page.getByRole('button', { name: 'Ajouter une personne' }).click()
  await page.getByRole('button', { name: 'Ajouter une personne' }).click()
  await page.getByRole('button', { name: 'Ajouter une personne' }).click()
  await page.getByLabel('a dépensé').click()
  await page.getByLabel('a dépensé').fill('')
  await page
    .getByText('DépenseurUn castor affairéUne autruche curieuseUn ornithorynque malicieux a dé')
    .click()
  await page.getByLabel('Une autruche curieuse').check()
  await page.getByLabel('Un ornithorynque malicieux').check()
  await page.getByRole('button', { name: 'Ajouter une dépense' }).click()
  await page.getByRole('button', { name: 'Ajouter une dépense' }).click()
  await page.getByLabel('a dépensé').click()
  await page.getByLabel('a dépensé').fill('34')
  await page
    .getByText('DépenseurUn castor affairéUne autruche curieuseUn ornithorynque malicieux a dé')
    .click()
  await page.locator('label').filter({ hasText: 'Un castor affairé' }).click()
  await page.getByRole('button', { name: 'Ajouter une dépense' }).click()
  await page.getByLabel('a dépensé').click()
  await page.getByLabel('a dépensé').fill('34.9')
  await page.getByRole('button', { name: 'Ajouter une dépense' }).click()
  await page.getByLabel('Dépenseur').selectOption('1')
  await page.getByLabel('Dépenseur').selectOption('2')
  await page.getByLabel('Dépenseur').selectOption('1')
})
