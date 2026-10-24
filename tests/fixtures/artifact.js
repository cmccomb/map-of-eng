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
    keyword_ids: [
      `keyword-${(index % 3) + 1}`,
      `keyword-${(index % 3) + 1}-${index < 6 ? Math.floor(index / 3) + 1 : 1}`,
    ],
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
    keyword_levels: [
      { level: 0, label: "Topic regions", keyword_count: 3 },
      { level: 1, label: "Detailed topics", keyword_count: 6 },
    ],
    keywords: [
      {
        keyword_id: "keyword-1",
        label: "robotic design",
        level: 0,
        parent_keyword_id: null,
        publication_count: 3,
        coordinates: {
          pca: { x: -0.1, y: 0.12 },
          tsne: { x: 0.1, y: -0.05 },
        },
      },
      {
        keyword_id: "keyword-2",
        label: "energy systems",
        level: 0,
        parent_keyword_id: null,
        publication_count: 3,
        coordinates: {
          pca: { x: 0.2, y: -0.1 },
          tsne: { x: -0.15, y: 0.18 },
        },
      },
      {
        keyword_id: "keyword-3",
        label: "biomedical interfaces",
        level: 0,
        parent_keyword_id: null,
        publication_count: 2,
        coordinates: {
          pca: { x: 0.45, y: -0.35 },
          tsne: { x: 0.58, y: 0.5 },
        },
      },
      ...[
        ["keyword-1-1", "robot control", "keyword-1", -0.5, 0.65],
        ["keyword-1-2", "design collaboration", "keyword-1", -0.12, 0.08],
        ["keyword-2-1", "soft systems", "keyword-2", 0.05, -0.2],
        ["keyword-2-2", "neural interfaces", "keyword-2", 0.56, -0.48],
        ["keyword-3-1", "battery health", "keyword-3", 0.72, 0.65],
        ["keyword-3-2", "power grid", "keyword-3", 0.82, 0.18],
      ].map(([keywordId, label, parentId, x, y]) => ({
        keyword_id: keywordId,
        label,
        level: 1,
        parent_keyword_id: parentId,
        publication_count: 1,
        coordinates: {
          pca: { x, y },
          tsne: { x: -x, y: -y },
        },
      })),
    ],
    points,
    additive_metadata: { accepted: true },
  };
}

function makeSchoolArtifact() {
  const artifact = makeArtifact();
  artifact.schema_version = 7;
  artifact.catalogs.schools = [
    { school_id: "engineering", title: "College of Engineering", publication_count: 9 },
    { school_id: "scs", title: "School of Computer Science", publication_count: 2 },
  ];
  artifact.catalogs.departments = artifact.catalogs.departments.map((item) => ({
    ...item,
    school_id: "engineering",
  }));
  artifact.catalogs.departments.push({
    department_id: "d-csd",
    title: "Computer Science Department",
    school_id: "scs",
    publication_count: 2,
  });
  artifact.catalogs.faculty = artifact.catalogs.faculty.map((item) => ({
    ...item,
    school_ids: ["engineering"],
  }));
  artifact.catalogs.faculty.push({
    person_id: "f-erin",
    display_name: "Erin Ellis",
    school_ids: ["scs"],
    publication_count: 2,
  });
  for (const point of artifact.points) point.school_ids = ["engineering"];
  artifact.points.push({
    ...makePoint(pointRows[0], 8),
    work_id: "p9",
    title: "Robotic Design Collaboration",
    authors: "Alice Adams; Erin Ellis",
    department_ids: ["d-me", "d-csd"],
    faculty_ids: ["f-alice", "f-erin"],
    school_ids: ["engineering", "scs"],
  });
  artifact.points.push({
    ...makePoint(pointRows[0], 9),
    work_id: "p10",
    title: "Computer Science Research",
    authors: "Erin Ellis",
    department_ids: ["d-csd"],
    faculty_ids: ["f-erin"],
    school_ids: ["scs"],
  });
  artifact.point_count = artifact.points.length;
  return artifact;
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

module.exports = { makeArtifact, makeSchoolArtifact, makeLargeArtifact };
