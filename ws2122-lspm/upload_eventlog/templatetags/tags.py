from django import template                                                                                                                                                        
from django.template.defaultfilters import stringfilter


register = template.Library()

@register.filter 
@stringfilter
def coma2double(s):
    return s.replace('&#039', '&quot;')
