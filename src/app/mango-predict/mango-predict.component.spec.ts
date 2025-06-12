import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MangoPredictComponent } from './mango-predict.component';

describe('MangoPredictComponent', () => {
  let component: MangoPredictComponent;
  let fixture: ComponentFixture<MangoPredictComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [MangoPredictComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MangoPredictComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
