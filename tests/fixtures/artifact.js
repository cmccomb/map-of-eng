"use strict";

const layouts = [
  {
    layout_id: "pca",
    label: "Global structure",
    method: "PCA",
    description: "Preserves broad relationships across the complete corpus.",
    x_field: "pca_x",
    y_field: "pca_y",
  },
  {
    layout_id: "tsne",
    label: "Local neighborhoods",
    method: "t-SNE",
    description: "Emphasizes compact groups of closely related publications.",
    x_field: "tsne_x",
    y_field: "tsne_y",
  },
];

const departments = [
  { department_id: "d-me", title: "Mechanical Engineering", publication_count: 5 },
  { department_id: "d-ece", title: "Electrical and Computer Engineering", publication_count: 2 },
  { department_id: "d-bme", title: "Biomedical Engineering", publication_count: 2 },
];

const faculty = [
  { person_id: "f-alice", display_name: "Alice Adams", publication_count: 3 },
  { person_id: "f-bob", display_name: "Bob Brown", publication_count: 3 },
  { person_id: "f-carol", display_name: "Carol Chen", publication_count: 2 },
  { person_id: "f-dan", display_name: "Dan Diaz", publication_count: 1 },
];

const pointRows = [
  ["p1", "Robotic Grasp Planning", "Alice Adams", ["d-me"], ["f-alice"], -0.8, 0.55, -0.2, 0.8],
  ["p2", "Soft Robot Control", "Alice Adams; Carol Chen", ["d-me", "d-bme"], ["f-alice", "f-carol"], -0.58, 0.38, 0.66, 0.6],
  ["p3", "Battery Health Prediction", "Bob Brown", ["d-ece"], ["f-bob"], 0.72, 0.65, -0.72, -0.62],
  ["p4", "Human-AI Design Collaboration", "Alice Adams; Bob Brown", ["d-me"], ["f-alice", "f-bob"], -0.12, 0.08, 0.12, 0.12],
  ["p5", "Neural Prosthetic Interface", "Carol Chen", ["d-bme"], ["f-carol"], 0.56, -0.48, 0.78, 0.72],
  ["p6", "Power Grid Optimization", "Bob Brown", ["d-ece"], ["f-bob"], 0.82, 0.18, -0.62, -0.78],
  ["p7", "Additive Manufacturing Process Design", "Dan Diaz", ["d-me"], ["f-dan"], -0.35, -0.72, 0.55, -0.48],
  ["p8", "Fluid Dynamics of Turbulent Jets", "External collaborator", ["d-me"], [], 0.18, -0.82, 0.24, -0.32],
];

function makePoint(row, index) {
  const [workId, title, authors, departmentIds, facultyIds, pcaX, pcaY, tsneX, tsneY] = row;
  return {
    work_id: workId,
    title,
    authors,
    department_ids: [...departmentIds],
    faculty_ids: [...facultyIds],
    pca_x: pcaX,
    pca_y: pcaY,
    tsne_x: tsneX,
    tsne_y: tsneY,
    year: 2018 + index,
    venue: `Journal ${index + 1}`,
    citation_count: index * 7,
    observation_count: index % 2 ? 2 : 1,
    doi: `10.1000/example.${index + 1}`,
    source_url: `https://example.org/publication/${index + 1}`,
    keyword_id: `keyword-${(index % 3) + 1}`,
  };
}

function makeArtifact() {
  const points = pointRows.map(makePoint);
  return {
    schema_version: 6,
    keyword_model_version: "fixture-keywords-v1",
    point_count: points.length,
    source_data_newest_at_utc: "2026-07-17T00:00:00Z",
    default_layout_id: "pca",
    layouts: layouts.map((layout) => ({ ...layout })),
    catalogs: {
      departments: departments.map((item) => ({ ...item })),
      faculty: faculty.map((item) => ({ ...item })),
    },
    keywords: [
      {
        keyword_id: "keyword-1",
        label: "robotic design",
        publication_count: 3,
        coordinates: {
          pca: { x: -0.1, y: 0.12 },
          tsne: { x: 0.1, y: -0.05 },
        },
      },
      {
        keyword_id: "keyword-2",
        label: "energy systems",
        publication_count: 3,
        coordinates: {
          pca: { x: 0.2, y: -0.1 },
          tsne: { x: -0.15, y: 0.18 },
        },
      },
      {
        keyword_id: "keyword-3",
        label: "biomedical interfaces",
        publication_count: 2,
        coordinates: {
          pca: { x: 0.45, y: -0.35 },
          tsne: { x: 0.58, y: 0.5 },
        },
      },
    ],
    points,
    additive_metadata: { accepted: true },
  };
}

function makeLargeArtifact(count = 32958) {
  const artifact = makeArtifact();
  artifact.points = Array.from({ length: count }, (_, index) => {
    const source = makePoint(pointRows[index % pointRows.length], index % pointRows.length);
    const angle = index * 2.399963229728653;
    const radius = Math.sqrt((index + 1) / count) * 0.96;
    return {
      ...source,
      work_id: `large-${index}`,
      pca_x: Math.cos(angle) * radius,
      pca_y: Math.sin(angle) * radius,
      tsne_x: Math.sin(angle * 1.7) * radius,
      tsne_y: Math.cos(angle * 1.3) * radius,
    };
  });
  artifact.point_count = count;
  return artifact;
}

module.exports = { makeArtifact, makeLargeArtifact };
