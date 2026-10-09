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
    authors: ['P. A. Massih', 'E. Cosatto'],
    title: 'Reasoning with Pixel-level Precision: QVLM Architecture and SQuID Dataset for Quantitative Geospatial Analytics',
    status: 'Preprint, arXiv:2601.13401, 2026. Dataset on Hugging Face.',
    summary: 'A benchmark of 2,000 quantitative questions on satellite images with tolerance ranges calibrated from human annotations. A code-generation baseline reaches 42.0% vs 28.1% for the best zero-shot VLM.',
    page: '/publications/qvlm/',
    links: [
      { label: 'arXiv', href: 'https://arxiv.org/abs/2601.13401' },
      { label: 'Dataset', href: 'https://huggingface.co/datasets/PeterAM4/SQuID' },
    ],
    bibtex: `@misc{massih2026squid,
  title         = {Reasoning with Pixel-level Precision: QVLM Architecture and SQuID Dataset for Quantitative Geospatial Analytics},
  author        = {Massih, Peter A. and Cosatto, Eric},
  year          = {2026},
  eprint        = {2601.13401},
  archivePrefix = {arXiv}
}`,
  },
  {
    authors: ['P. A. Massih', 'E. Cosatto'],
    title: 'What Is a Visual Thought Worth? When Continuous Visual States Help Vision-Language Models Reason',
    status: 'Under review at ICLR 2027.',
    summary: 'Tests whether the latent visual states of CoVT and LVR can be decoded, are used and add value beyond image features, by removing the image while keeping the states fixed.',
  },
  {
    authors: ['S. Lu', 'P. A. Massih', 'E. Cosatto', 'M. R. Min'],
    title: 'Test-Time Reasoning with Computational Evidence for Spatial Question Answering',
    status: 'Under review at ICLR 2027.',
    summary: 'A training-free framework that plans question-specific geometric computations and gives the VLM explicit spatial evidence for multi-hop spatial questions.',
  },
  {
    authors: ['P. A. Massih'],
    title: 'Three Modalities of Production AI in Financial Services: Retrieval, Generation, and Detection',
    status: "Master's thesis, EPFL, 2026. Open access on Zenodo.",
    summary: 'Deployed 3 AI systems under FINMA rules, used daily by bankers, fund screeners and tax specialists, and managed their releases, CI/CD, unit tests and Kafka integration.',
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
