/*
 * Copyright 2025 Thousand Brains Project
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT license
 * that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

const DOCS_BASE_URL = 'https://thousandbrainsproject.readme.io/docs/';
const GITHUB_EDIT_BASE_URL = 'https://github.com/thousandbrainsproject/tbp.monty/edit/main/';
const GITHUB_AVATAR_URL = 'https://github.com';
const EXTERNAL_LINK_ICON = 'fa-external-link-alt';
const EDIT_ICON = 'fa-pencil-alt';
const BADGE_CLASS = 'badge';
const BADGE_SKILLS_CLASS = 'badge-skills';
const BADGE_STATUS_CLASS = 'badge-status';


function escapeHtml(unsafe) {
  if (unsafe == null) return '';
  return he.encode(String(unsafe));
}


function addToSearch(value) {
  const input = document.getElementById('searchInput');
  const currentValue = input.value.trim();
  const searchTerm = value.trim();

  if (currentValue.includes(searchTerm)) {
    input.value = currentValue.replace(searchTerm, '').replace(/\s+/g, ' ').trim();
  } else {
    input.value = currentValue ? `${currentValue} ${searchTerm}` : searchTerm;
  }

  input.dispatchEvent(new Event('input', { bubbles: true }));
}


function updateUrlSearchParam(searchTerm) {
  const url = new URL(window.location);

  const trimmedSearchTerm = searchTerm.trim();
  if (trimmedSearchTerm) {
    url.searchParams.set('q', trimmedSearchTerm);
  } else {
    url.searchParams.delete('q');
  }

  window.history.replaceState({}, '', url);
}


function getInitialSearchTerm() {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get('q') || '';
}


const ColumnFormatters = {

  formatArrayOrStringColumn(value, cssClass) {
    const items = Array.isArray(value)
      ? value.filter(Boolean)
      : (value || '').split(',').map(item => item.trim()).filter(Boolean);

    return items
      .map(item => `<span class="${escapeHtml(cssClass)}" data-search-value="${escapeHtml(item)}" style="cursor: pointer;">${escapeHtml(item)}</span>`)
      .join(' ');
  },
  formatLinkColumn(cell, icon = EXTERNAL_LINK_ICON, urlPrefix = '') {
    const value = cell.getValue();
    if (!value) return '';

    const url = urlPrefix ? `${urlPrefix}${value}` : value;
    return `<a href="${url}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(url)}"><i class="${icon}"></i></a>`;
  },
  formatTagsColumn: (cell) => ColumnFormatters.formatArrayOrStringColumn(cell.getValue(), BADGE_CLASS),
  formatSkillsColumn: (cell) => ColumnFormatters.formatArrayOrStringColumn(cell.getValue(), BADGE_SKILLS_CLASS),
  formatSizeColumn(cell) {
    const value = (cell.getValue() || '').trim().toLowerCase();
    return value
      ? `<span class="badge badge-size-${escapeHtml(value)}" data-search-value="${escapeHtml(value)}" style="cursor: pointer;">${escapeHtml(value)}</span>`
      : '';
  },
  formatSlugLinkColumn: (cell) => ColumnFormatters.formatLinkColumn(cell, EXTERNAL_LINK_ICON, DOCS_BASE_URL),
  formatEditLinkColumn: (cell) => ColumnFormatters.formatLinkColumn(cell, EDIT_ICON, GITHUB_EDIT_BASE_URL),
  formatTitleWithLinksColumn(cell) {
    const rowData = cell.getRow().getData();
    const title = cell.getValue() || '';
    const slug = rowData.slug || '';
    const path = rowData.path || '';

    let result = escapeHtml(title);

    if (slug) {
      const docsUrl = `${DOCS_BASE_URL}${slug}`;
      result = `<a href="${escapeHtml(docsUrl)}" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: inherit;">${escapeHtml(title)}</a>`;
    }

    if (path) {
      const editUrl = `${GITHUB_EDIT_BASE_URL}${path}`;
      result = `<a href="${escapeHtml(editUrl)}" style="margin-right:5px;" target="_blank" rel="noopener noreferrer" title="Edit on GitHub"><i class="fas ${EDIT_ICON}"></i></a>${result}`;
    }

    return `<div style="margin-right: 10px; word-wrap: break-word; overflow-wrap: break-word;">${result}</div>`;
  },
  formatStatusColumn(cell) {
    const rowData = cell.getRow().getData();
    const status = cell.getValue() || '';
    const contributor = rowData.contributor || '';

    const statusText = status
      ? `<span class="${BADGE_STATUS_CLASS}">${escapeHtml(status)}</span>`
      : '';

    if (!contributor) return statusText;

    const usernames = Array.isArray(contributor)
      ? contributor
      : contributor.split(',').map(u => u.trim()).filter(Boolean);

    const avatars = usernames
      .map(username => `<img src="${GITHUB_AVATAR_URL}/${encodeURIComponent(username)}.png" class="github-avatar" data-search-value="${escapeHtml(username)}" title="${escapeHtml(username)}" alt="${escapeHtml(username)}"/>`)
      .join(' ');

    return statusText + '<br>' + avatars;
  },
  formatRfcColumn(cell) {
    const value = cell.getValue();
    if (!value) return '';

    const isHttpUrl = /^https?:/.test(value.trim());
    return isHttpUrl
      ? `<a href="${escapeHtml(value)}" target="_blank" rel="noopener noreferrer">RFC <i class="fas ${EXTERNAL_LINK_ICON}"></i></a>`
      : escapeHtml(value);
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
      { title: 'Title', field: 'title', formatter: ColumnFormatters.formatTitleWithLinksColumn, width: 200, cssClass: 'wrap-text', variableHeight: true },
      { title: 'Scope', field: 'estimated-scope', formatter: ColumnFormatters.formatSizeColumn },
      { title: 'Metric', field: 'improved-metric', formatter: ColumnFormatters.formatTagsColumn, maxWidth: 200, cssClass: 'wrap-text' },
      { title: 'Output Type', field: 'output-type', formatter: ColumnFormatters.formatTagsColumn },
      { title: 'RFC', field: 'rfc', formatter: ColumnFormatters.formatRfcColumn },
      { title: 'Status', field: 'status', formatter: ColumnFormatters.formatStatusColumn },
      { title: 'Tags', field: 'tags', formatter: ColumnFormatters.formatTagsColumn, widthGrow: 2, cssClass: 'wrap-text' },
      { title: 'Skills', field: 'skills', formatter: ColumnFormatters.formatSkillsColumn, widthGrow: 2, cssClass: 'wrap-text' }
    ];
  },


  getDisplayColumns() {
    const allColumns = this.getAllColumns();
    const columnsToShow = this.getColumnsToShow();

    if (!columnsToShow) {
      return allColumns;
    }

    const columnMap = new Map(
      allColumns.map(col => [col.field.toLowerCase(), col])
    );

    return columnsToShow
      .map(fieldName => columnMap.get(fieldName))
      .filter(Boolean);
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
      this.showError('Failed to load data - see the console for more details or refresh the page to try again.');
    }
  },


  async loadData() {
    const response = await fetch('data.json', {
      cache: 'no-store'
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}, body: ${await response.text()}`);
    }
    const data = await response.json();
    return data.slice().sort((a, b) =>
      (a.path2 || '').localeCompare(b.path2 || '', undefined, { sensitivity: 'base' }) ||
      (a.title || '').localeCompare(b.title || '', undefined, { sensitivity: 'base' })
    );
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
    const clearLink = document.getElementById('clearSearch');
    const copyUrlLink = document.getElementById('copyUrl');
    const hideCompletedCheckbox = document.getElementById('hideCompleted');

    const applyFilters = () => {
      const searchTerm = searchInput.value.toLowerCase().trim();
      const hideCompleted = hideCompletedCheckbox.checked;

      if (!searchTerm && !hideCompleted) {
        table.clearFilter();
        return;
      }

      table.setFilter((data) => {
        if (hideCompleted && (data.status || '').toLowerCase() === 'completed') {
          return false;
        }

        if (!searchTerm) {
          return true;
        }

        const searchableText = [
          data.title, data.tags, data.skills, data.status,
          data.contributor, data['estimated-scope'], data['improved-metric'],
          data['output-type'], data.rfc, data.link, data.path2
        ]
          .filter(Boolean)
          .join(' ')
          .toLowerCase();

        return searchTerm.split(/\s+/).every(word => searchableText.includes(word));
      });
    };

    const initialSearchTerm = getInitialSearchTerm();
    if (initialSearchTerm) {
      searchInput.value = initialSearchTerm;
    }
    applyFilters();

    searchInput.addEventListener('input', (e) => {
      updateUrlSearchParam(e.target.value);
      applyFilters();
    });

    hideCompletedCheckbox.addEventListener('change', applyFilters);

    if (clearLink) {
      clearLink.addEventListener('click', (e) => {
        e.preventDefault();
        searchInput.value = '';
        updateUrlSearchParam('');
        applyFilters();
      });
    }

    if (copyUrlLink) {
      copyUrlLink.addEventListener('click', async (e) => {
        e.preventDefault();
        await this.handleCopyUrl(copyUrlLink);
      });
    }

    document.addEventListener('click', (e) => {
      if (e.target.dataset.searchValue) {
        addToSearch(e.target.dataset.searchValue);
      }
    });
  },


  async handleCopyUrl(element) {
    const setTemporaryState = (icon, className = null) => {
      element.textContent = icon;
      if (className) element.classList.add(className);
      setTimeout(() => {
        element.textContent = '📋';
        if (className) element.classList.remove(className);
      }, 1500);
    };

    try {
      await navigator.clipboard.writeText(window.location.href);
      setTemporaryState('✅', 'success');
    } catch (err) {
      console.error('Failed to copy URL to clipboard:', err);
      setTemporaryState('❌');
    }
  },


  showError(message) {
    const tableElement = document.getElementById('table');
    if (tableElement) {
      tableElement.innerHTML = `<div class="error-message" style="color: red; padding: 20px; text-align: center; font-weight: bold;">${escapeHtml(message)}</div>`;
    }
  }
};


document.addEventListener('DOMContentLoaded', () => {
  FutureWorkWidget.init();
});
