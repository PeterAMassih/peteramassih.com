// src/data/publications.ts
// The four papers in a fixed order, shared by the home page and /publications/.
// Entries with a page also live in src/content/publications/. Papers under
// review get no link, page or BibTeX until they are public.

export interface Pub {
  authors: string[];
  title: string;
  status: string;
  summary?: string;
  page?: string;
  links?: { label: string; href: string }[];
  bibtex?: string;
}

// How his name is printed on the papers, set in bold in author lists.
export const ME = 'P. A. Massih';

export const pubs: Pub[] = [
  {
    authors: ['P. A. Massih', 'E. Cosatto', 'M. R. Min'],
    title: 'SQuID: A Benchmark for Quantitative Spatial Reasoning on Satellite Imagery',
    status: 'Preprint, arXiv:2601.13401, 2026',
    summary: 'SQuID, a benchmark of 2,000 quantitative questions on satellite images with tolerance ranges informed by human annotations. The best zero-shot end-to-end VLM reaches 28.1%, and the QASR baseline reaches 42.0%.',
    page: '/publications/squid/',
    links: [
      { label: 'arXiv', href: 'https://arxiv.org/abs/2601.13401' },
      { label: 'Dataset', href: 'https://huggingface.co/datasets/PeterAM4/SQuID' },
      { label: 'Code', href: 'https://github.com/PeterAMassih/SQuID' },
    ],
    bibtex: `@misc{massih2026squid,
  title         = {{SQuID}: A Benchmark for Quantitative Spatial Reasoning on Satellite Imagery},
  author        = {Massih, Peter A. and Cosatto, Eric and Min, Martin Renqiang},
  year          = {2026},
  eprint        = {2601.13401},
  archivePrefix = {arXiv}
}`,
  },
  {
    authors: ['P. A. Massih', 'E. Cosatto'],
    title: 'What Is a Visual Thought Worth? When Continuous Visual States Help Vision-Language Models Reason',
    status: 'Under review at ICLR 2027',
    summary: 'Removes the image and keeps the latent visual states of CoVT and LVR fixed, to test whether the states can be decoded, are used, and add value beyond image features.',
  },
  {
    authors: ['S. Lu', 'P. A. Massih', 'E. Cosatto', 'M. R. Min'],
    title: 'Test-Time Reasoning with Computational Evidence for Spatial Question Answering',
    status: 'Under review at ICLR 2027',
    summary: 'A training-free framework that plans question-specific geometric computations and gives the VLM explicit computational evidence for multi-hop spatial questions.',
  },
  {
    authors: ['P. A. Massih'],
    title: 'Three Modalities of Production AI in Financial Services: Retrieval, Generation, and Detection',
    status: "Master's thesis, EPFL, 2026",
    summary: 'Three AI systems deployed under FINMA rules and used daily by bankers, fund screeners and tax specialists.',
    page: '/publications/msc-thesis/',
    links: [
      { label: 'DOI', href: 'https://doi.org/10.5281/zenodo.21074981' },
      { label: 'PDF', href: '/master-thesis.pdf' },
    ],
    bibtex: `@mastersthesis{massih2026thesis,
  title  = {Three Modalities of Production AI in Financial Services: Retrieval, Generation, and Detection},
  author = {Massih, Peter A.},
  school = {EPFL},
  year   = {2026},
  doi    = {10.5281/zenodo.21074981}
}`,
  },
];
