# TokenSqueeze Demo Prompts

These are natural, paragraph-length prompts designed to showcase TokenSqueeze's intent detection. Despite being long and conversational, each is fundamentally a binary yes/no question - meaning we can use low detail mode (85 tokens) instead of high detail (~1,000+ tokens).

---

## 1. Receipt Verification (BINARY_QUESTION) - ~89% savings

**Prompt:**
> "I just got back from dinner with my team and I need to submit this receipt for expense reimbursement. Before I do, can you quickly check - is the total amount on this receipt more than $50? My company requires manager approval for anything over that threshold."

**Image to download:** Search for "restaurant receipt high resolution" - get a clear photo of a restaurant receipt with itemized items, ideally showing a total around $40-60.

**Why it works:** Long natural prompt, but it's fundamentally a yes/no question. BINARY_QUESTION intent → low detail mode (85 tokens).

---

## 2. Product Authenticity Check (BINARY_QUESTION) - ~89% savings

**Prompt:**
> "I bought this Nike sneaker from an online marketplace and I'm worried it might be a counterfeit. Looking at the label on the inside of the shoe, does the stitching and logo placement look legitimate to you? I want to know if I should request a refund before the return window closes."

**Image to download:** Search for "Nike shoe label inside tongue" or "sneaker authenticity label" - get a close-up of a shoe's interior label.

**Why it works:** Emotional, relatable scenario. Still a binary question (legit or not).

---

## 3. Parking Sign Confusion (BINARY_QUESTION) - ~89% savings

**Prompt:**
> "I'm parked on this street and I'm trying to understand if I'm going to get a ticket. It's currently 2pm on a Tuesday. Based on all the signs in this photo, am I legally parked right now or do I need to move my car immediately?"

**Image to download:** Search for "confusing parking signs multiple" or "no parking sign complex" - get an image with multiple overlapping parking restriction signs.

**Why it works:** Relatable urban problem, complex image but simple yes/no answer needed.

---

## 4. Medication Verification (BINARY_QUESTION) - ~89% savings

**Prompt:**
> "My elderly mother accidentally mixed up her pill bottles and I need to verify something before she takes her evening medication. Looking at this pill bottle label, is this the correct medication - Lisinopril 10mg? I want to make sure she doesn't take the wrong one."

**Image to download:** Search for "prescription pill bottle label" - get a clear photo of a prescription medication label.

**Why it works:** High-stakes scenario that judges can relate to (healthcare costs), simple binary verification.

---

## Token Savings Breakdown

| Scenario | Original Tokens | Optimized Tokens | Savings |
|----------|----------------|------------------|---------|
| Receipt Verification | ~1,000+ | 85 | ~89% |
| Product Authenticity | ~1,000+ | 85 | ~89% |
| Parking Sign | ~1,000+ | 85 | ~89% |
| Medication Verification | ~1,000+ | 85 | ~89% |

**Key Insight:** All four prompts are paragraph-length and sound complex, but TokenSqueeze correctly identifies them as binary questions that only need a yes/no answer. This means we can use GPT-4o's "low" detail mode which costs a flat 85 tokens regardless of image size.

---

## Why These Prompts Work for Demo

1. **Relatable scenarios** - Expense reports, online shopping, parking, healthcare
2. **Natural language** - Not obviously "yes/no" questions
3. **High-resolution images** - Maximizes the "before" token count
4. **Same answer** - Model gives identical response with 89% fewer tokens
5. **Real cost impact** - Easy to calculate monthly savings at scale
