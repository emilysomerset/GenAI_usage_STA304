
# ============================================================
# Generative AI Usage in STA304
# Summary table and figure
# ============================================================

library(dplyr)
library(tidyr)
library(ggplot2)
library(kableExtra)

# Data are available through a restricted-access Zenodo repository:
# https://doi.org/10.5281/zenodo.23376068

# load("../Clean_data/data_clean.RData")


# ============================================================
# 1. Generative AI Usage by Term
# ============================================================

# Order in which terms appear in the summary table
term_order <- c("Fall 2023", "Winter 2025", "Fall 2025")

# Responses indicating that GenAI was used on at least one assignment
usage_responses <- c(
  "I used generative AI for Assignment 1, but not Assignment 2.",
  "I used generative AI for Assignment 2, but not Assignment 1.",
  "I used generative AI for both Assignment 1 and Assignment 2"
)

# Add an aggregate category for students who used GenAI on
# at least one assignment. Retain the original responses so
# both the aggregate and individual categories can be reported.
usage_data <- work_d %>%
  filter(genai_usage2 %in% usage_responses) %>%
  mutate(genai_usage2 = "At least one assignment") %>%
  bind_rows(work_d)


# Calculate counts and percentages by term.
#
# For the main response categories, the denominator is the
# total number of non-missing responses within each term.
#
# For the three individual GenAI usage categories, the
# denominator is the number of students who used GenAI
# on at least one assignment.
make_summary <- function(data, var) {
  
  data %>%
    filter(!is.na(.data[[var]])) %>%
    count(term, response = .data[[var]]) %>%
    group_by(term) %>%
    mutate(
      n_any = n[response == "At least one assignment"][1],
      n_total = sum(n) - n_any,
      denom = if_else(
        response %in% usage_responses,
        n_any,
        n_total
      ),
      pct = round(100 * n / denom, 1),
      cell = paste0(pct, "\\% (n=", n, ")")
    ) %>%
    ungroup() %>%
    select(response, term, cell)
}


# Reshape the summary so that terms appear as columns.
usage_summary <- make_summary(usage_data, "genai_usage2") %>%
  pivot_wider(
    names_from = term,
    values_from = cell,
    values_fill = "—"
  )


# Define the response labels and their order in the table.
response_order <- c(
  "At least one assignment",
  "I used generative AI for Assignment 1, but not Assignment 2.",
  "I used generative AI for Assignment 2, but not Assignment 1.",
  "I used generative AI for both Assignment 1 and Assignment 2",
  "Neither assignment",
  "Do not know"
)

response_labels <- c(
  "\\hspace{1em}At least one assignment",
  "\\hspace{2em}Assignment 1 only",
  "\\hspace{2em}Assignment 2 only",
  "\\hspace{2em}Both assignments",
  "\\hspace{1em}Neither assignment",
  "\\hspace{1em}Do not know"
)


# Prepare the table, including a heading row.
table_data <- usage_summary %>%
  mutate(
    response = factor(response, levels = response_order)
  ) %>%
  arrange(response) %>%
  mutate(Response = response_labels[as.integer(response)]) %>%
  select(Response, all_of(term_order))

heading_row <- tibble(
  Response = "Usage of Generative AI on STA304 assignments",
  !!!setNames(as.list(rep("", length(term_order))), term_order)
)

table_data <- bind_rows(heading_row, table_data)


# Create the LaTeX table.
genai_usage_table <- table_data %>%
  kbl(
    format = "latex",
    booktabs = TRUE,
    caption = "Generative AI usage by term",
    align = "lccc",
    escape = FALSE,
    label = "tab:genai-summary",
    col.names = c("", term_order)
  ) %>%
  kable_styling(
    latex_options = c("hold_position", "scale_down"),
    font_size = 10
  ) %>%
  row_spec(1, bold = TRUE)

genai_usage_table


# ============================================================
# 2. Generative AI Tasks by Term
# ============================================================

# Summarize reported GenAI use for specific tasks.
#
# Restrict the analysis to:
#   - Students with a1 > 0 and a2 > 0
#   - Winter 2025 and Fall 2025
#
# Each task is summarized separately. Percentages are
# calculated among students with non-missing responses
# for that task within each term.

genai_task_summary <- work_d %>%
  filter(
    a1 > 0,
    a2 > 0,
    term %in% c("Winter 2025", "Fall 2025")
  ) %>%
  pivot_longer(
    cols = starts_with("genai_task_"),
    names_to = "task",
    values_to = "response"
  ) %>%
  filter(!is.na(response)) %>%
  mutate(
    task = recode(
      task,
      genai_task_code_scratch = "Code (from scratch)",
      genai_task_code_edit = "Code (editing)",
      genai_task_text_scratch = "Text (from scratch)",
      genai_task_text_edit = "Text (editing)",
      genai_task_formulas = "Formulas",
      genai_task_citations = "Citations"
    ),
    response = factor(
      response,
      levels = c(
        "neither assignment",
        "Assignment 1 only",
        "Assignment 2 only",
        "both assignments"
      )
    )
  ) %>%
  count(term, task, response) %>%
  group_by(term, task) %>%
  mutate(pct = 100 * n / sum(n)) %>%
  ungroup()


# Retain the four coding and writing tasks for the figure.
# Specify task order so coding tasks appear above writing tasks
# after the coordinate axes are flipped.
task_order <- rev(c(
  "Code (editing)",
  "Code (from scratch)",
  "Text (editing)",
  "Text (from scratch)"
))

genai_task_summary <- genai_task_summary %>%
  filter(!task %in% c("Citations", "Formulas")) %>%
  mutate(
    task = factor(task, levels = task_order),
    term = factor(term, levels = c("Winter 2025", "Fall 2025"))
  )


# ------------------------------------------------------------
# 2.1 Stacked bar chart
# ------------------------------------------------------------

# Each bar shows the percentage of students reporting
# GenAI use for a given task on neither, one, or both assignments.
#
# Percentage labels are displayed only for segments >= 8%
# to avoid overcrowding the figure.

gg <- ggplot(
  genai_task_summary,
  aes(x = task, y = pct, fill = response)
) +
  geom_col(width = 0.72) +
  
  # White labels on the darkest segments
  geom_text(
    aes(label = ifelse(
      response == "both assignments" & pct >= 8,
      paste0(round(pct), "%"),
      ""
    )),
    position = position_stack(vjust = 0.5),
    color = "white",
    fontface = "bold",
    size = 3.3
  ) +
  
  # Dark labels on the remaining segments
  geom_text(
    aes(label = ifelse(
      response != "both assignments" & pct >= 8,
      paste0(round(pct), "%"),
      ""
    )),
    position = position_stack(vjust = 0.5),
    color = "grey20",
    size = 3.3
  ) +
  
  facet_wrap(~term, nrow = 1) +
  coord_flip() +
  
  # Colours and legend ordering
  scale_fill_manual(
    values = c(
      "neither assignment" = "grey85",
      "Assignment 1 only" = "#9ecae1",
      "Assignment 2 only" = "#4292c6",
      "both assignments" = "#08519c"
    ),
    breaks = c(
      "both assignments",
      "Assignment 2 only",
      "Assignment 1 only",
      "neither assignment"
    ),
    labels = c(
      "Both assignments",
      "Assignment 2 only",
      "Assignment 1 only",
      "Neither"
    )
  ) +
  
  scale_y_continuous(
    breaks = seq(0, 100, 25),
    labels = scales::label_percent(scale = 1),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  
  labs(
    x = NULL,
    y = "Percent of students",
    fill = NULL
  ) +
  
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    strip.text = element_text(face = "bold", size = 12),
    axis.text.y = element_text(size = 11),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.spacing.x = grid::unit(1.2, "lines"),
    plot.margin = margin(10, 15, 10, 10)
  )

gg


# ------------------------------------------------------------
# 2.2 Export figure
# ------------------------------------------------------------

ggsave(
  filename = "../Plots/task_usage.pdf",
  plot = gg,
  width = 8,
  height = 4,
  units = "in"
)

# Open the exported PDF in the RStudio viewer.
rstudioapi::viewer("../Plots/task_usage.pdf")


# ============================================================
# 3. Other Generative AI Uses: Open-Ended Responses
# ============================================================

# Export non-missing open-ended responses from Winter 2025
# and Fall 2025 for qualitative coding.
work_d %>%
  filter(
    term %in% c("Winter 2025", "Fall 2025"),
    !is.na(genai_other_tasks)
  ) %>%
  select(genai_other_tasks) %>%
  write.csv(
    file = "../eecs_workflow/free_text_genai_use.csv",
    row.names = FALSE
  )
