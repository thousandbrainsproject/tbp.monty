
const CONSTANTS = {
  URLs: {
    DOCS_BASE: 'https://thousandbrainsproject.readme.io/docs/',
    GITHUB_EDIT_BASE: 'https://github.com/thousandbrainsproject/tbp.monty/edit/main/',
    GITHUB_AVATAR: 'https://github.com'
  },
  ICONS: {
    EXTERNAL_LINK: 'fa-external-link-alt',
    EDIT: 'fa-pencil-alt'
  },
  CSS_CLASSES: {
    BADGE: 'badge',
    BADGE_SKILLS: 'badge-skills',
    WRAP_TEXT: 'wrap-text'
  },
  STATUS: {
    IN_PROGRESS: 'in-progress'
  }
};


const SearchManager = {

  addToSearch(value) {
    const input = document.getElementById('searchInput');
    const words = input.value.trim().split(/\s+/).filter(Boolean);
    const index = words.indexOf(value);

    if (index !== -1) {
      words.splice(index, 1);
    } else {
      words.push(value);
    }

    input.value = words.join(' ');
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }
};


const ColumnFormatters = {

  formatArrayOrStringColumn(value, cssClass) {
    const items = Array.isArray(value)
      ? value.filter(Boolean)
      : (value || '').split(',').map(item => item.trim()).filter(Boolean);

    return items
      .map(item => `<span class="${cssClass}" onclick="SearchManager.addToSearch('${item.replace(/'/g, "\\'")}')">${item}</span>`)
      .join(' ');
  },
  formatLinkColumn(cell, icon = CONSTANTS.ICONS.EXTERNAL_LINK, urlPrefix = '') {
    const value = cell.getValue();
    if (!value) return '';

    const url = urlPrefix ? `${urlPrefix}${value}` : value;
    return `<a href="${url}" target="_blank" rel="noopener noreferrer" title="${url}"><i class="fas ${icon}"></i></a>`;
  },
  formatTagsColumn: (cell) => ColumnFormatters.formatArrayOrStringColumn(cell.getValue(), CONSTANTS.CSS_CLASSES.BADGE),
  formatSkillsColumn: (cell) => ColumnFormatters.formatArrayOrStringColumn(cell.getValue(), CONSTANTS.CSS_CLASSES.BADGE_SKILLS),
  formatSizeColumn(cell) {
    const value = (cell.getValue() || '').trim().toLowerCase();
    return value
      ? `<span class="badge badge-size-${value}" onclick="SearchManager.addToSearch('${value}')">${value}</span>`
      : '';
  },
  formatSlugLinkColumn: (cell) => ColumnFormatters.formatLinkColumn(cell, CONSTANTS.ICONS.EXTERNAL_LINK, CONSTANTS.URLs.DOCS_BASE),
  formatEditLinkColumn: (cell) => ColumnFormatters.formatLinkColumn(cell, CONSTANTS.ICONS.EDIT, CONSTANTS.URLs.GITHUB_EDIT_BASE),
  formatTitleWithLinksColumn(cell) {
    const rowData = cell.getRow().getData();
    const title = cell.getValue() || '';
    const slug = rowData.slug || '';
    const path = rowData.path || '';

    let result = title;

    if (slug) {
      const docsUrl = `${CONSTANTS.URLs.DOCS_BASE}${slug}`;
      result = `<a href="${docsUrl}" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: inherit;">${title}</a>`;
    }

    if (path) {
      const editUrl = `${CONSTANTS.URLs.GITHUB_EDIT_BASE}${path}`;
      result = `<a href="${editUrl}" style="margin-right:5px;" target="_blank" rel="noopener noreferrer" title="Edit on GitHub"><i class="fas ${CONSTANTS.ICONS.EDIT}"></i></a>${result}`;
    }

    return `<div style="margin-right: 10px;">${result}</div>`;
  },
  formatStatusColumn(cell) {
    const rowData = cell.getRow().getData();
    const status = cell.getValue() || '';
    const owner = rowData.owner || '';

    if (!owner) return status;

    const usernames = Array.isArray(owner)
      ? owner
      : owner.split(',').map(u => u.trim()).filter(Boolean);

    const avatars = usernames
      .map(username => `<img src="${CONSTANTS.URLs.GITHUB_AVATAR}/${encodeURIComponent(username)}.png"
                             width="16" height="16"
                             style="vertical-align:middle;border-radius:2px;margin-left:5px;"
                             alt="${username}"/>`)
      .join(' ');

    return status.toLowerCase() === CONSTANTS.STATUS.IN_PROGRESS
      ? avatars
      : status + avatars;
  },
  formatRfcColumn(cell) {
    const value = cell.getValue();
    if (!value) return '';

    const isHttpUrl = /^https?:/.test(value.trim());
    return isHttpUrl
      ? `<a href="${value}" target="_blank" rel="noopener noreferrer">RFC <i class="fas ${CONSTANTS.ICONS.EXTERNAL_LINK}"></i></a>`
      : value;
  }
};


const TableConfig = {

  getColumnsToShow() {
    const urlParams = new URLSearchParams(window.location.search);
    const columnsParam = urlParams.get('columns');

    return columnsParam
      ? columnsParam.split(',').map(col => col.trim().toLowerCase())
      : null;
  },


  getAllColumns() {
    return [
      { title: 'Title', field: 'title', formatter: ColumnFormatters.formatTitleWithLinksColumn },
      { title: 'Estimated Scope', field: 'estimated-scope', formatter: ColumnFormatters.formatSizeColumn },
      { title: 'RFC', field: 'rfc', formatter: ColumnFormatters.formatRfcColumn },
      { title: 'Status', field: 'status', formatter: ColumnFormatters.formatStatusColumn },
      { title: 'Tags', field: 'tags', formatter: ColumnFormatters.formatTagsColumn, widthGrow: 2, cssClass: CONSTANTS.CSS_CLASSES.WRAP_TEXT },
      { title: 'Skills', field: 'skills', formatter: ColumnFormatters.formatSkillsColumn, widthGrow: 2, cssClass: CONSTANTS.CSS_CLASSES.WRAP_TEXT }
    ];
  },


  getDisplayColumns() {
    const allColumns = this.getAllColumns();
    const columnsToShow = this.getColumnsToShow();

    return columnsToShow
      ? allColumns.filter(col =>
          columnsToShow.includes(col.field.toLowerCase()) ||
          columnsToShow.includes(col.title.toLowerCase())
        )
      : allColumns;
  }
};


const FutureWorkWidget = {

  async init() {
    try {
      const data = await this.loadData();
      const table = this.createTable(data);
      this.setupSearch(table);
    } catch (error) {
      console.error('Failed to initialize Future Work Widget:', error);
      this.showError('Failed to load data. Please refresh the page.');
    }
  },


  async loadData() {
    const response = await fetch('data.json');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return Array.isArray(data)
      ? data.slice().sort((a, b) => {
          const pathA = (a.path2 || '').toLowerCase();
          const pathB = (b.path2 || '').toLowerCase();
          if (pathA < pathB) return -1;
          if (pathA > pathB) return 1;
          const titleA = (a.title || '').toLowerCase();
          const titleB = (b.title || '').toLowerCase();
          if (titleA < titleB) return -1;
          if (titleA > titleB) return 1;
          return 0;
        })
      : data;
  },


  createTable(data) {
    return new Tabulator('#table', {
      data: data,
      layout: 'fitDataStretch',
      columns: TableConfig.getDisplayColumns(),
      groupBy: 'path2'
    });
  },


  setupSearch(table) {
    const searchInput = document.getElementById('searchInput');
    if (!searchInput) {
      console.warn('Search input not found');
      return;
    }

    searchInput.addEventListener('input', (e) => {
      const searchTerm = e.target.value.toLowerCase().trim();

      if (!searchTerm) {
        table.clearFilter();
        return;
      }

      table.setFilter((data) => {
        const searchableText = [
          data.title, data.tags, data.skills, data.status,
          data.owner, data['estimated-scope'], data.link, data.path2
        ]
          .filter(Boolean)
          .join(' ')
          .toLowerCase();

        return searchTerm
          .split(/\s+/)
          .every(word => searchableText.includes(word));
      });
    });
  },


  showError(message) {
    const errorElement = document.createElement('div');
    errorElement.className = 'error-message';
    errorElement.textContent = message;
    errorElement.style.cssText = 'color: red; padding: 20px; text-align: center; font-weight: bold;';

    const tableElement = document.getElementById('table');
    if (tableElement) {
      tableElement.appendChild(errorElement);
    }
  }
};


window.SearchManager = SearchManager;


document.addEventListener('DOMContentLoaded', () => {
  FutureWorkWidget.init();
});
