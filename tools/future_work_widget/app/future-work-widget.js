function addToSearch(value) {
  const input = document.getElementById("searchInput");
  const words = input.value.trim().split(/\s+/).filter(Boolean);
  const index = words.indexOf(value);
  if (index !== -1) words.splice(index, 1);
  else words.push(value);
  input.value = words.join(" ");
  input.dispatchEvent(new Event('input', { bubbles: true }));
}

function formatArrayOrStringColumn(value, cssClass) {
  const items = Array.isArray(value) ? value.filter(Boolean) :
    (value || "").split(",").map(item => item.trim()).filter(Boolean);
  return items.map(item =>
    `<span class="${cssClass}" onclick="addToSearch('${item.replace(/'/g, "\\'")}')">${item}</span>`
  ).join(" ");
}

function formatLinkColumn(cell, icon = "fa-external-link-alt", urlPrefix = "") {
  const value = cell.getValue();
  if (!value) return "";
  const url = urlPrefix ? `${urlPrefix}${value}` : value;
  return `<a href="${url}" target="_blank" rel="noopener noreferrer" title="${url}"><i class="fas ${icon}"></i></a>`;
}

const formatTagsColumn = (cell) => formatArrayOrStringColumn(cell.getValue(), "badge");
const formatSkillsColumn = (cell) => formatArrayOrStringColumn(cell.getValue(), "badge-skills");
const formatSizeColumn = (cell) => {
  const value = (cell.getValue() || "").trim().toLowerCase();
  return value ? `<span class="badge badge-size-${value}" onclick="addToSearch('${value}')">${value}</span>` : "";
};
const formatSlugLinkColumn = (cell) => formatLinkColumn(cell, "fa-external-link-alt", "https://thousandbrainsproject.readme.io/docs/");
const formatEditLinkColumn = (cell) => formatLinkColumn(cell, "fa-pencil-alt", "https://github.com/thousandbrainsproject/tbp.monty/edit/main/");
const formatTitleWithLinksColumn = (cell) => {
  const rowData = cell.getRow().getData();
  const title = cell.getValue() || "";
  const slug = rowData.slug || "";
  const path = rowData.path || "";

  let result = title;

  if (slug) {
    const docsUrl = `https://thousandbrainsproject.readme.io/docs/${slug}`;
    result = `<a href="${docsUrl}" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: inherit;">${title}</a>`;
  }

  if (path) {
    const editUrl = `https://github.com/thousandbrainsproject/tbp.monty/edit/main/${path}`;
    result = ` <a href="${editUrl}" style="margin-right:5px;" target="_blank" rel="noopener noreferrer"  title="Edit on GitHub"><i class="fas fa-pencil-alt"></i></a>` + result;
  }

  return '<div style="margin-right: 10px;">' + result + '</div>';
};
const formatStatusColumn = (cell) => {
  const rowData = cell.getRow().getData();
  const status = cell.getValue() || "";
  const owner = rowData.owner || "";

  let result = status;

  if (owner) {
    const usernames = Array.isArray(owner) ? owner : owner.split(",").map(u => u.trim()).filter(Boolean);
    const avatars = usernames.map(username =>
      `<img src="https://github.com/${encodeURIComponent(username)}.png" width="24" height="24" style="vertical-align:middle;border-radius:2px;margin-left:5px;" alt="${username}"/>`
    ).join(" ");

    if (status.toLowerCase() === "in-progress") {
      result = avatars;
    } else {
      result = status + avatars;
    }
  }

  return result;
};

function getColumnsToShow() {
  const urlParams = new URLSearchParams(window.location.search);
  const columnsParam = urlParams.get('columns');

  if (columnsParam) {
    return columnsParam.split(',').map(col => col.trim().toLowerCase());
  }

  return null;
}

const allColumns = [
  { title: "Title", field: "title", formatter: formatTitleWithLinksColumn },
  { title: "Estimated Scope", field: "estimated-scope", formatter: formatSizeColumn },
  { title: "RFC", field: "rfc" },
  { title: "Status", field: "status", formatter: formatStatusColumn },
  { title: "Tags", field: "tags", formatter: formatTagsColumn, widthGrow: 2, cssClass: "wrap-text" },
  { title: "Skills", field: "skills", formatter: formatSkillsColumn, widthGrow: 2, cssClass: "wrap-text" }
];

const columnsToShow = getColumnsToShow();
const displayColumns = columnsToShow ?
  allColumns.filter(col => columnsToShow.includes(col.field.toLowerCase()) || columnsToShow.includes(col.title.toLowerCase())) :
  allColumns;

fetch('data.json')
  .then(res => res.json())
  .then(data => {
    const table = new Tabulator("#table", {
      data: data,
      layout: 'fitDataStretch',
      initialSort: [{ column: "title", dir: "asc" }],
      columns: displayColumns,
      groupBy: "path2"
    });

    document.getElementById("searchInput").addEventListener("input", e => {
      const searchTerm = e.target.value.toLowerCase().trim();
      if (!searchTerm) {
        table.clearFilter();
      } else {
        table.setFilter(data => {
          const allText = [data.title, data.tags, data.skills, data.status, data.owner, data["estimated-scope"], data.link, data.path2]
            .filter(Boolean).join(" ").toLowerCase();
          return searchTerm.split(/\s+/).every(word => allText.includes(word));
        });
      }
    });
  })
  .catch(console.error);
